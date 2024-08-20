#include "brick.h"

#define CHECK_CUDNN(err)                                            \
  do {                                                               \
    if (CUDNN_STATUS_SUCCESS != err) {                               \
      std::cerr << "cuDNN Error - " << cudnnGetErrorString(err) << std::endl; \
      exit(1);                                                        \
    }                                                               \
  } while (0)

#define CHECK_CUDA(err)                                            \
  do {                                                               \
    if (cudaSuccess != err) {                                       \
      std::cerr << "CUDA Error - " << cudaGetErrorString(err) << std::endl; \
      exit(1);                                                        \
    }                                                               \
  } while (0)


struct TensorDims {
  int N, C, H, W;
};


cudnnStream_t createStream() {
  cudnnStream_t stream;
  CHECK_CUDNN(cudnnCreateStream(&stream));
  return stream;
}


void destroyStream(cudnnStream_t stream) {
  CHECK_CUDNN(cudnnDestroyStream(stream));
}

void* allocateTensor(cudnnDataType_t dataType, const TensorDims& dims) {
  size_t dataSize = dims.N * dims.C * dims.H * dims.W * cudnnSizeOf(dataType);
  void* dataPtr;
  CHECK_CUDA(cudaMalloc(&dataPtr, dataSize));
  return dataPtr;
}

void freeTensor(void* ptr) {
  CHECK_CUDA(cudaFree(ptr));
}

cudnnTensorDescriptor_t createTensorDesc(cudnnDataType_t dataType, const TensorDims& dims) {
  cudnnTensorDescriptor_t desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
  CHECK_CUDNN(cudnnSetTensorDescriptor(desc, CUDNN_TENSOR_NCHW, dataType, dims.N, dims.C, dims.H, dims.W));
  return desc;
}

void destroyTensorDesc(cudnnTensorDescriptor_t desc) {
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
}

class ResNet50Layer {
public:
  virtual ~ResNet50Layer() = default;

  virtual void forward(const float* input, float* output, cudnnStream_t stream = nullptr) = 0;
};

class ConvBatchnormLayer : public ResNet50Layer {
public:
  ConvBatchnormLayer(int inChannels, int outChannels, int kernelSize, int stride, bool addIdentity = false);

  void forward(const float* input, float* output, cudnnStream_t stream = nullptr) override;

private:
  cudnnFilterDescriptor_t filterDesc_;
  cudnnConvolutionDescriptor_t convDesc_;
  cudnnTensorDescriptor_t biasDesc_;
  float* weights_;
  float* bias_;
  cudnnBatchNormDescriptor_t batchnormDesc_;
  float* runningMean_;
  float* runningVar_;
  float* eps_;
  cudnnActivationDescriptor_t activationDesc_;
};

ConvBatchnormLayer::ConvBatchnormLayer(int inChannels, int outChannels, int kernelSize, int stride, bool addIdentity) {
  // Allocate memory for weights, bias, running mean, running variance, epsilon
  weights_ = reinterpret_cast<float*>(allocateTensor(CUDNN_DATA_FLOAT, {1, outChannels, inChannels, kernelSize, kernelSize}));
  bias_ = reinterpret_cast<float*>(allocateTensor(CUDNN_DATA_FLOAT, {1, outChannels, 1, 1}));
  runningMean_ = reinterpret_cast<float*>(allocateTensor(CUDNN_DATA_FLOAT, {1, outChannels, 1, 1}));
  runningVar_ = reinterpret_cast<float*>(allocateTensor(CUDNN_DATA_
