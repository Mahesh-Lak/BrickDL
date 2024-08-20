#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

// Helper function to check for CUDA errors
#define CUDA_CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Helper function to check for cuDNN errors
#define CUDNN_CHECK(call) \
{ \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Set tensor dimensions
    const int batchSize = 1;
    const int channels = 3;
    const int depth = 224;
    const int height = 224;
    const int width = 224;

    // Allocate device memory for input tensor
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, batchSize * channels * depth * height * width * sizeof(float)));

    // Create tensor descriptor
    cudnnTensorDescriptor_t inputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, channels, depth, height * width));

    // Define convolution parameters
    const int kernelDepth = 3;
    const int kernelHeight = 3;
    const int kernelWidth = 3;
    const int padDepth = 1;
    const int padHeight = 1;
    const int padWidth = 1;
    const int strideDepth = 1;
    const int strideHeight = 1;
    const int strideWidth = 1;

    // Create convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution3dDescriptor(convDesc, padDepth, padHeight, padWidth, strideDepth, strideHeight, strideWidth, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Allocate device memory for convolution filters and biases
    float* d_filter[6];
    float* d_bias[6];
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMalloc(&d_filter[i], channels * channels * kernelDepth * kernelHeight * kernelWidth * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias[i], channels * sizeof(float)));
    }

    // Perform six 3D convolution operations
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < 3; ++i) {
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, convDesc, d_filter[i], CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, inputDesc, d_input));
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, inputDesc, d_input, &alpha, inputDesc, d_bias[i]));
        CUDNN_CHECK(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, d_input, &beta, inputDesc, d_input));
    }

    // Destroy cuDNN descriptors and handle
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    for (int i = 0; i < 6; ++i) {
        CUDA_CHECK(cudaFree(d_filter[i]));
        CUDA_CHECK(cudaFree(d_bias[i]));
    }

    return 0;
}
