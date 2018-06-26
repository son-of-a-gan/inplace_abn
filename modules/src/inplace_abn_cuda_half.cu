#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <vector>

#include "common.h"
#include "inplace_abn.h"

// Operations for reduce
struct SumOpH {
  __device__ SumOpH(const half *t, int c, int s)
      : tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    return __half2float(tensor[(batch * chn + plane) * sp + n]);
  }
  const half *tensor;
  const int chn;
  const int sp;
};

struct VarOpH {
  __device__ VarOpH(float m, const half *t, int c, int s)
      : mean(m), tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    const auto t = __half2float(tensor[(batch * chn + plane) * sp + n]);
    return (t - mean) * (t - mean);
  }
  const float mean;
  const half *tensor;
  const int chn;
  const int sp;
};

/***********
 * mean_var
 ***********/

__global__ void mean_var_kernel_h(const half *x, float *mean, float *var, int num, int chn, int sp) {
  int plane = blockIdx.x;
  float norm = 1.f / static_cast<float>(num * sp);

  float _mean = reduce<float, SumOpH>(SumOpH(x, chn, sp), plane, num, chn, sp) * norm;
  __syncthreads();
  float _var = reduce<float, VarOpH>(VarOpH(_mean, x, chn, sp), plane, num, chn, sp) * norm;

  if (threadIdx.x == 0) {
    mean[plane] = _mean;
    var[plane] = _var;
  }
}

std::vector<at::Tensor> mean_var_cuda_h(at::Tensor x) {
  CHECK_INPUT(x);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto mean = at::empty(x.type().toScalarType(at::kFloat), {chn});
  auto var = at::empty(x.type().toScalarType(at::kFloat), {chn});

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  mean_var_kernel_h<<<blocks, threads>>>(
      reinterpret_cast<half*>(x.data<at::Half>()),
      mean.data<float>(),
      var.data<float>(),
      num, chn, sp);

  return {mean, var};
}

/**********
 * forward
 **********/

__global__ void forward_kernel_h(half *x, const float *mean, const float *var, const float *weight, const float *bias,
                                 bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  const float _mean = mean[plane];
  const float _var = var[plane];
  const float _weight = affine ? abs(weight[plane]) + eps : 1.f;
  const float _bias = affine ? bias[plane] : 0.f;

  const float mul = rsqrt(_var + eps) * _weight;

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      half *x_ptr = x + (batch * chn + plane) * sp + n;
      const float _x = __half2float(*x_ptr);
      const float _y = (_x - _mean) * mul + _bias;

      *x_ptr = __float2half(_y);
    }
  }
}

at::Tensor forward_cuda_h(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(mean);
  CHECK_INPUT(var);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  forward_kernel_h<<<blocks, threads>>>(
      reinterpret_cast<half*>(x.data<at::Half>()),
      mean.data<float>(),
      var.data<float>(),
      weight.data<float>(),
      bias.data<float>(),
      affine, eps, num, chn, sp);

  return x;
}