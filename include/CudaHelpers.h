#pragma once

#include <cuda_runtime.h>
#include <iostream>
namespace CudaHelpers {
    // Helper function to get cores per SM based on compute capability
    int _ConvertSMVer2Cores(int major, int minor) {
        // Refer to NVIDIA documentation or SDK examples for a comprehensive list
        // This is a simplified example and may not cover all architectures
        switch (major) {
            case 2: // Fermi
                if (minor == 1) return 48;
                return 32;
            case 3: // Kepler
                return 192;
            case 5: // Maxwell
                return 128;
            case 6: // Pascal
                if (minor == 0) return 64;
                return 128;
            case 7: // Volta, Turing
                return 64;
            case 8: // Ampere
                if (minor == 0) return 64; // A100
                return 128; // Other Ampere
            case 9: // Ada Lovelace
                return 128;
            default:
                return 0; // Unknown or unsupported
        }
    }

    void checkCudaErrors() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }

    void printDeviceInfo() {
        int device = 0;
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        std::cout << "GPU name: " << props.name << "\n";
        std::cout << "Streaming Multiprocessors (SMs): " << props.multiProcessorCount << "\n";
        std::cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
        std::cout << "Number of CUDA Cores: " << props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor) << "\n";
    }
}
