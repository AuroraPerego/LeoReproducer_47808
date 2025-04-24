#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
// PixelTrackUtilities only included in order to compile SoALayout with Eigen columns
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"

#include "RecoVertex/PixelVertexFinding/plugins/PixelVertexWorkSpaceUtilities.h"
#include "RecoVertex/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoAHost.h"
#include "RecoVertex/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoADevice.h"
#include "RecoVertex/PixelVertexFinding/plugins/gpuClusterTracksByDensity.h"
#include "RecoVertex/PixelVertexFinding/plugins/gpuFitVertices.h"
#include "RecoVertex/PixelVertexFinding/plugins/gpuSortByPt2.h"
#include "RecoVertex/PixelVertexFinding/plugins/gpuSplitVertices.h"

struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t> itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<float> pttrack;
  std::vector<uint16_t> ivert;
};

struct ClusterGenerator {
  explicit ClusterGenerator(float nvert, float ntrack)
      : rgen(-13., 13), errgen(0.005, 0.025), clusGen(nvert), trackGen(ntrack), gauss(0., 1.), ptGen(1.) {}

  void operator()(Event& ev) {
    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto& z : ev.zvert) {
      z = 3.5f * gauss(reng);
    }

    ev.ztrack.clear();
    ev.eztrack.clear();
    ev.ivert.clear();
    ev.pttrack.clear();
    for (int iv = 0; iv < nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[iv] = nt;
      for (int it = 0; it < nt; ++it) {
        auto err = errgen(reng);  // reality is not flat....
        ev.ztrack.push_back(ev.zvert[iv] + err * gauss(reng));
        ev.eztrack.push_back(err * err);
        ev.ivert.push_back(iv);
        ev.pttrack.push_back((iv == 5 ? 1.f : 0.5f) + ptGen(reng));
        ev.pttrack.back() *= ev.pttrack.back();
      }
    }
    // add noise
    auto nt = 2 * trackGen(reng);
    for (int it = 0; it < nt; ++it) {
      auto err = 0.03f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err * err);
      ev.ivert.push_back(9999);
      ev.pttrack.push_back(0.5f + ptGen(reng));
      ev.pttrack.back() *= ev.pttrack.back();
    }
  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::uniform_real_distribution<float> errgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;
  std::exponential_distribution<float> ptGen;
};

__global__ void print(gpuVertexFinder::VtxSoAView pdata, gpuVertexFinder::WsSoAView pws) {
  auto& __restrict__ ws = pws;
  printf("nt,nv %d %d,%d\n", ws.ntrks(), pdata.nvFinal(), ws.nvIntermediate());
}

int main() {
  cudaStream_t stream;
  cms::cudatest::requireDevices();
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  ZVertexSoADevice onGPU_d(stream);
  gpuVertexFinder::workSpace::PixelVertexWorkSpaceSoADevice ws_d(stream);

  Event ev;

  float eps = 0.1f;
  std::array<float, 3> par{{eps, 0.01f, 9.0f}};
  for (int nav = 30; nav < 80; nav += 20) {
    ClusterGenerator gen(nav, 10);

    for (int i = 8; i < 20; ++i) {
      auto kk = i / 4;  // M param

      gen(ev);

      gpuVertexFinder::init<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      auto nt = ev.ztrack.size();
      cudaCheck(cudaMemcpy(&ws_d.view().ntrks(), &nt, sizeof(uint32_t), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().zt(), ev.ztrack.data(), sizeof(float) * ev.ztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().ezt2(), ev.eztrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(
          cudaMemcpy(ws_d.view().ptt2(), ev.pttrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));

      std::cout << "M eps, pset " << kk << ' ' << eps << ' ' << (i % 4) << std::endl;

      if ((i % 4) == 0)
        par = {{eps, 0.02f, 12.0f}};
      if ((i % 4) == 1)
        par = {{eps, 0.02f, 9.0f}};
      if ((i % 4) == 2)
        par = {{eps, 0.01f, 9.0f}};
      if ((i % 4) == 3)
        par = {{0.7f * eps, 0.01f, 9.0f}};

      uint32_t nv = 0;
      print<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());
      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

      cms::cuda::launch(gpuVertexFinder::clusterTracksByDensityKernel,
                        {1, 512 + 256},
                        onGPU_d.view(),
                        ws_d.view(),
                        kk,
                        par[0],
                        par[1],
                        par[2]);
      print<<<1, 1, 0, stream>>>(onGPU_d.view(), ws_d.view());

      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 50.f);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      float* zv = nullptr;
      float* wv = nullptr;
      float* ptv2 = nullptr;
      int32_t* nn = nullptr;
      uint16_t* ind = nullptr;

      // keep chi2 separated...
      float chi2[2 * nv];  // make space for splitting...

      float hzv[2 * nv];
      float hwv[2 * nv];
      float hptv2[2 * nv];
      int32_t hnn[2 * nv];
      uint16_t hind[2 * nv];

      zv = hzv;
      wv = hwv;
      ptv2 = hptv2;
      nn = hnn;
      ind = hind;

      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 50.f);
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "before splitting nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      // one vertex per block!!!
      cms::cuda::launch(gpuVertexFinder::splitVerticesKernel, {1024, 64}, onGPU_d.view(), ws_d.view(), 9.f);
      cudaCheck(cudaMemcpy(&nv, &ws_d.view().nvIntermediate(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
      std::cout << "after split " << nv << std::endl;

      cms::cuda::launch(gpuVertexFinder::fitVerticesKernel, {1, 1024 - 256}, onGPU_d.view(), ws_d.view(), 5000.f);
      cudaCheck(cudaGetLastError());

      cms::cuda::launch(gpuVertexFinder::sortByPt2Kernel, {1, 256}, onGPU_d.view(), ws_d.view());
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, &onGPU_d.view().nvFinal(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      cudaCheck(cudaMemcpy(zv, onGPU_d.view().zv(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(wv, onGPU_d.view().wv(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, onGPU_d.view().chi2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ptv2, onGPU_d.view().ptv2(), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, onGPU_d.view().ndof(), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ind, onGPU_d.view().sortInd(), nv * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      {
        auto mx = std::minmax_element(wv, wv + nv);
        std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second) << std::endl;
      }

      {
        auto mx = std::minmax_element(ptv2, ptv2 + nv);
        std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
        std::cout << "min max ptv2 " << ptv2[ind[0]] << ' ' << ptv2[ind[nv - 1]] << " at " << ind[0] << ' '
                  << ind[nv - 1] << std::endl;
      }

      float dd[nv];
      for (auto kv = 0U; kv < nv; ++kv) {
        auto zr = zv[kv];
        auto md = 500.0f;
        for (auto zs : ev.ztrack) {
          auto d = std::abs(zr - zs);
          md = std::min(d, md);
        }
        dd[kv] = md;
      }
      if (i == 6) {
        for (auto d : dd)
          std::cout << d << ' ';
        std::cout << std::endl;
      }
      auto mx = std::minmax_element(dd, dd + nv);
      float rms = 0;
      for (auto d : dd)
        rms += d * d;
      rms = std::sqrt(rms) / (nv - 1);
      std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;

    }  // loop on events
  }  // lopp on ave vert

  return 0;
}
