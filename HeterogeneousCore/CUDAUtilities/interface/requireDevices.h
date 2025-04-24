#ifndef HeterogeneousCore_CUDAUtilities_requireDevices_h
#define HeterogeneousCore_CUDAUtilities_requireDevices_h

#include <cuda_runtime.h>

/**
 * These functions are meant to be called only from unit tests.
 */
namespace cms {
  namespace cudatest {
    /// In presence of CUDA devices, return true; otherwise print message and return false
    bool testDevices() {
      int devices = 0;
      auto status = cudaGetDeviceCount(&devices);
      if (status != cudaSuccess) {
        std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped."
                  << "\n";
        return false;
      }
      if (devices == 0) {
        std::cerr << "No CUDA devices available, the test will be skipped."
                  << "\n";
        return false;
      }
      return true;
    }

    /// Print message and exit if there are no CUDA devices
    void requireDevices() {
      if (not testDevices()) {
        exit(EXIT_SUCCESS);
      }
    }
  }  // namespace cudatest
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_requireDevices_h
