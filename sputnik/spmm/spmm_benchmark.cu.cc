// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <cstdint>
#include <functional>

#include "glog/logging.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include "benchmark/benchmark.h"
#include "absl/random/random.h"

namespace sputnik {

void LogBenchmarkArguments(const benchmark::State& state) {
  // Print the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "m=" << state.range(0) << ","
            << "k=" << state.range(1) << ","
            << "n=" << state.range(2) << ","
            << "nonzeros=" << state.range(3);
}

typedef std::function<cudaError_t(
    int,            // m: number of rows in lhs & output.
    int,            // k: number of cols in lhs and rows in rhs.
    int,            // n: number of cols in rhs/output.
    int,            // nonzeros: number of nonzero values in lhs.
    const int*,     // row_indices: ptr to row index swizzle map.
    const float*,   // values: ptr to lhs values.
    const int*,     // row_offsets: ptr to lhs row offsets.
    const int*,     // column_indices: ptr to lhs column indices.
    const float*,   // dense_matrix: ptr to rhs matrix.
    const float*,   // bias: ptr to bias.
    float*,         // output_matrix: ptr to output matrix.
    cudaStream_t)>  // stream: stream to execute in.
    FloatSpmmFn;

void BenchmarkFn(FloatSpmmFn spmm_fn, benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  // With the addition of the reverse offset memory alignment trick we
  // no longer need to pad rows of the sparse matrix to use vector memory
  // instruction.
  const int kRowPadding = 0;

  // Round the batch size so we can use the vector kernels.
  int aligned_ndim = kDimN;
  if ((kDimN % 4) != 0) {
    aligned_ndim = (kDimN + 3) / 4 * 4;
  }

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> sparse_matrix_gpu(
      kDimM, kDimK, kNonZeros, RANDOM_UNIFORM, &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix_gpu(kDimK, aligned_ndim, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix_gpu(kDimM, aligned_ndim, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(spmm_fn(kDimM, kDimK, aligned_ndim,
                        sparse_matrix_gpu.NumElementsWithPadding(),
                        sparse_matrix_gpu.RowIndices(),
                        sparse_matrix_gpu.Values(),
                        sparse_matrix_gpu.RowOffsets(),
                        sparse_matrix_gpu.ColumnIndices(), matrix_gpu.Values(),
                        /* bias = */ nullptr, output_matrix_gpu.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // log the benchmark arguments for parsing
  LogBenchmarkArguments(state);
}

typedef std::function<cudaError_t(
    int,            // m: number of rows in lhs & output.
    int,            // k: number of cols in lhs and rows in rhs.
    int,            // n: number of cols in rhs/output.
    int,            // nonzeros: number of nonzero values in lhs.
    const int*,     // row_indices: ptr to row index swizzle map.
    const half2*,   // values: ptr to lhs values.
    const int*,     // row_offsets: ptr to lhs row offsets.
    const short2*,  // column_indices: ptr to lhs column indices.
    const half2*,   // dense_matrix: ptr to rhs matrix.
    const float*,   // bias: ptr to bias.
    half2*,         // output_matrix: ptr to output matrix.
    cudaStream_t)>  // stream: stream to execute in.
    HalfSpmmFn;

void BenchmarkFn(HalfSpmmFn spmm_fn, benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  // Pad to the nearest 2 nonzero elements so we can use the `half2`
  // datatype.
  const int kRowPadding = 2;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<half2> sparse_matrix(kDimM, kDimK, kNonZeros, RANDOM_UNIFORM,
                                        &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<half2> matrix(kDimK, kDimN, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<half2> output_matrix(kDimM, kDimN, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(spmm_fn(kDimM, kDimK, kDimN,
                        sparse_matrix.NumElementsWithPadding(),
                        sparse_matrix.RowIndices(), sparse_matrix.Values(),
                        sparse_matrix.RowOffsets(),
                        sparse_matrix.ColumnIndices(), matrix.Values(),
                        /* bias = */ nullptr, output_matrix.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // log the benchmark arguments for parsing
  LogBenchmarkArguments(state);
}

void BenchmarkArgs(testing::Benchmark* b) {
  const std::vector<std::vector<int>> benchmarks_14 = {
      {89, 48, 12544, 427},   {176, 89, 3136, 1566},   {176, 176, 3136, 3098},
      {360, 176, 784, 6336},  {360, 360, 784, 12960},  {720, 360, 196, 25920},
      {720, 720, 196, 51840}, {1432, 720, 49, 103105}, {1432, 1432, 49, 205064},
  };

  const std::vector<std::vector<int>> benchmarks_18 = {
      {115, 56, 12544, 644},  {232, 115, 3136, 2668},  {232, 232, 3136, 5382},
      {464, 232, 784, 10765}, {464, 464, 784, 21530},  {920, 464, 196, 42688},
      {920, 920, 196, 84640}, {1840, 920, 49, 169280}, {1840, 1840, 49, 338560},
  };

  const std::vector<std::vector<int>> benchmarks_22 = {
      {140, 72, 12544, 1008},    {280, 140, 3136, 3920},
      {280, 280, 3136, 7840},    {560, 280, 784, 15680},
      {560, 560, 784, 31360},    {1128, 560, 196, 63169},
      {1128, 1128, 196, 127239}, {2256, 1128, 49, 254479},
      {2256, 2256, 49, 508919},
  };

  for (const auto& a : benchmarks_18) {
    b->Args(a);
  }
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ANONYMOUS_NAME(x) CONCAT(x, __COUNTER__)

#define REGISTER_BENCHMARK(name, fn)                                  \
  void BM_##name(benchmark::State& state) { BenchmarkFn(fn, state); } \
  BENCHMARK(BM_##name)->Apply(BenchmarkArgs)

#define REGISTER_FLOAT_BENCHMARK_HELPER(name, tname, fn, stype, dtype, mt, kt, \
                                        nt, bs)                                \
  const auto& tname = fn<SpmmConfig<float, stype, dtype, mt, kt, nt, bs>>;     \
  REGISTER_BENCHMARK(name##_##stype##x##dtype##x##mt##x##kt##x##nt##x##bs,     \
                     tname);

#define REGISTER_FLOAT_BENCHMARK(name, fn, stype, dtype, mt, kt, nt, bs)    \
  REGISTER_FLOAT_BENCHMARK_HELPER(name, ANONYMOUS_NAME(spmm_fn), fn, stype, \
                                  dtype, mt, kt, nt, bs);

/* 1-d tiling with blocksize 64 */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float, float, 1, 32, 64, 32);

/* 2-d tiling with blocksize 64 and vector loads */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float2, float2, 2, 32, 64, 16);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float4, 4, 32, 64, 8);

/* 1-d tilings with blocksize 32 */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float, float, 1, 32, 32, 32);

/* 2-d tilings with 32 n-dim and vector loads */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float2, float2, 2, 32, 32, 16);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float4, 4, 32, 32, 8);

/* 2-d tilings with 16 n-dim and vector loads */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float2, float2, 4, 32, 16, 8);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float2, float, 2, 32, 16, 16);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float4, 8, 32, 16, 4);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float2, 4, 32, 16, 8);

/* 2-d tilings with 8 n-dim and vector loads */
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float2, float2, 8, 32, 8, 4);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float4, 16, 32, 8, 2);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float2, 8, 32, 8, 4);
REGISTER_FLOAT_BENCHMARK(CudaSpmmEx, CudaSpmmEx, float4, float, 4, 32, 8, 8);

#undef REGISTER_FLOAT_BENCHMARK
#undef REGISTER_FLOAT_BENCHMARK_HELPER
#undef REGISTER_BENCHMARK

#define REGISTER_BENCHMARK(name, fn)                                  \
  void BM_##name(benchmark::State& state) { BenchmarkFn(fn, state); } \
  BENCHMARK(BM_##name)->Apply(BenchmarkArgs)

#define REGISTER_HALF_BENCHMARK_HELPER(name, tname, fn, stype, dtype, mt, kt, \
                                       nt, bs)                                \
  const auto& tname = fn<SpmmConfig<half2, stype, dtype, mt, kt, nt, bs>>;    \
  REGISTER_BENCHMARK(name##_##stype##x##dtype##x##mt##x##kt##x##nt##x##bs,    \
                     tname);

#define REGISTER_HALF_BENCHMARK(name, fn, stype, dtype, mt, kt, nt, bs)    \
  REGISTER_HALF_BENCHMARK_HELPER(name, ANONYMOUS_NAME(spmm_fn), fn, stype, \
                                 dtype, mt, kt, nt, bs);

/* 1-d tiling with blocksize 64 */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half2, half2, 1, 32, 64, 32);

/* 2-d tiling with blocksize 64 and vector loads */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half4, half4, 2, 32, 64, 16);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half8, 4, 32, 64, 8);

/* 1-d tilings with blocksize 32 */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half2, half2, 1, 32, 32, 32);

/* 2-d tilings with 32 n-dim and vector loads */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half4, half4, 2, 32, 32, 16);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half8, 4, 32, 32, 8);

/* 2-d tilings with 16 n-dim and vector loads */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half4, half4, 4, 32, 16, 8);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half4, half2, 2, 32, 16, 16);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half8, 8, 32, 16, 4);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half4, 4, 32, 16, 8);

/* 2-d tilings with 8 n-dim and vector loads */
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half4, half4, 8, 32, 8, 4);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half8, 16, 32, 8, 2);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half4, 8, 32, 8, 4);
REGISTER_HALF_BENCHMARK(CudaSpmmEx, CudaSpmmEx, half8, half2, 4, 32, 8, 8);

#undef ANONYMOUS_NAME
#undef CONCAT
#undef CONCAT_

void BM_CudaSpmm_Generic(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  // With the addition of the reverse offset memory alignment trick we
  // no longer need to pad rows of the sparse matrix to use vector memory
  // instruction.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> sparse_matrix_gpu(
      kDimM, kDimK, kNonZeros, RANDOM_UNIFORM, &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix_gpu(kDimK, kDimN, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix_gpu(kDimM, kDimN, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(CudaSpmm(
          kDimM, kDimK, kDimN, sparse_matrix_gpu.NumElementsWithPadding(),
          sparse_matrix_gpu.RowIndices(), sparse_matrix_gpu.Values(),
          sparse_matrix_gpu.RowOffsets(), sparse_matrix_gpu.ColumnIndices(),
          matrix_gpu.Values(), output_matrix_gpu.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // log the benchmark arguments for parsing
  LogBenchmarkArguments(state);
}

BENCHMARK(BM_CudaSpmm_Generic)->Apply(BenchmarkArgs);

void BM_CusparseSpmm(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  // Do not pad the rows for our CuSparse benchmarks.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> sparse_matrix_gpu(kDimM, kDimK, kNonZeros,
                                            RANDOM_UNIFORM, &generator,
                                            IDENTITY, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix_gpu(kDimK, kDimN, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix_gpu(kDimM, kDimN, &generator);

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;
  cusparseMatDescr_t mdesc;
  CUSPARSE_CALL(cusparseCreateMatDescr(&mdesc));
  CUSPARSE_CALL(cusparseSetMatIndexBase(mdesc, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(cusparseSetMatType(mdesc, CUSPARSE_MATRIX_TYPE_GENERAL));

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // NOTE: CuSparse csrmm expects the left-hand side sparse matrix to be
      // stored in compressed sparse row format, the right-hand side dense
      // matrix to be stored in column-major format, and the output dense
      // matrix to be stored in column-major format. However, CuSparse
      // provides a second routine "csrmm2" which can take a row-major
      // right-hand side dense matrix to improve memory access patterns.
      // We do this by passing in a row-major right-hand side matrix and
      // setting Op(B) = B^T. The output is column-major, which is different
      // from our kernel.
      //
      // TODO(tgale): We should also benchmark the standard csrmm kernel
      // with both dense matrices stored in column-major.
      CUSPARSE_CALL(cusparseScsrmm2(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_TRANSPOSE, kDimM, kDimN, kDimK, kNonZeros, &alpha,
          mdesc, sparse_matrix_gpu.Values(), sparse_matrix_gpu.RowOffsets(),
          sparse_matrix_gpu.ColumnIndices(), matrix_gpu.Values(), kDimN, &beta,
          output_matrix_gpu.Values(), kDimM));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  LogBenchmarkArguments(state);
}

BENCHMARK(BM_CusparseSpmm)->Apply(BenchmarkArgs);

void GemmBenchmarkArgs(testing::Benchmark* b) {
  const std::vector<std::vector<int>> benchmarks_10 = {
      {64, 32, 12544, 2048},     {128, 64, 3136, 8192},
      {128, 128, 3136, 16384},   {256, 128, 784, 32768},
      {256, 256, 784, 65536},    {512, 256, 196, 131072},
      {512, 512, 196, 262144},   {1024, 512, 49, 524288},
      {1024, 1024, 49, 1048576},
  };

  const std::vector<std::vector<int>> benchmarks_14 = {
      {89, 48, 12544, 4272},     {176, 89, 3136, 15664},
      {176, 176, 3136, 30976},   {360, 176, 784, 63360},
      {360, 360, 784, 129600},   {720, 360, 196, 259200},
      {720, 720, 196, 518400},   {720, 720, 196, 518400},
      {720, 720, 196, 518400},   {720, 720, 196, 518400},
      {720, 720, 196, 518400},   {1432, 720, 49, 1031040},
      {1432, 1432, 49, 2050624},
  };

  const std::vector<std::vector<int>> benchmarks_18 = {
      {115, 56, 12544, 6440},    {232, 115, 3136, 26680},
      {232, 232, 3136, 53824},   {464, 232, 784, 107648},
      {464, 464, 784, 215296},   {920, 464, 196, 426880},
      {920, 920, 196, 846400},   {1840, 920, 49, 1692800},
      {1840, 1840, 49, 3385600},
  };

  for (const auto& a : benchmarks_14) {
    b->Args(a);
  }
}

void BM_CublasColumnMajorGemm(benchmark::State& state) {
  BenchmarkUseRealTime();
  int m = state.range(0);
  int k = state.range(1);
  int n = state.range(2);

  // Create the lhs, rhs, and output matrices on gpu.
  absl::BitGen generator;
  CudaMatrix<float> lhs_gpu(m, k, &generator);
  CudaMatrix<float> rhs_gpu(k, n, &generator);
  CudaMatrix<float> output_gpu(m, n, &generator);

  // Setup CuBLAS specific data structures.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;
  cudaDataType_t data_type = CUDA_R_32F;
  cudaDataType_t compute_type = CUDA_R_32F;
  cublasGemmAlgo_t gemm_algorithm = CUBLAS_GEMM_DEFAULT;

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // Run the cublas kernel.
      CUBLAS_CALL(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, lhs_gpu.Values(),
          data_type, m, rhs_gpu.Values(), data_type, k, &beta,
          output_gpu.Values(), data_type, m, compute_type, gemm_algorithm));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }
  LogBenchmarkArguments(state);
}

BENCHMARK(BM_CublasColumnMajorGemm)->Apply(GemmBenchmarkArgs);

cublasStatus_t RowMajorGemm(cublasHandle_t handle, bool trans_a,
                            const CudaMatrix<float>& a, bool trans_b,
                            const CudaMatrix<float>& b, CudaMatrix<float>* c) {
  int m = trans_b ? b.Rows() : b.Columns();
  int k = trans_b ? b.Columns() : b.Rows();
  int n = trans_a ? a.Columns() : a.Rows();

  int ldb = trans_b ? k : m;
  int lda = trans_a ? n : k;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;
  return cublasGemmEx(handle, transpose_b, transpose_a, m, n, k, &alpha,
                      b.Values(), CUDA_R_32F, ldb, a.Values(), CUDA_R_32F, lda,
                      &beta, c->Values(), CUDA_R_32F, c->Columns(), CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
}

void BM_CublasRowMajorGemm(benchmark::State& state) {
  BenchmarkUseRealTime();
  int m = state.range(0);
  int k = state.range(1);
  int n = state.range(2);

  // Create the lhs, rhs, and output matrices on gpu.
  absl::BitGen generator;
  CudaMatrix<float> lhs_gpu(m, k, &generator);
  CudaMatrix<float> rhs_gpu(k, n, &generator);
  CudaMatrix<float> output_gpu(m, n, &generator);

  // Setup CuBLAS specific data structures.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // Run the cublas kernel.
      CUBLAS_CALL(
          RowMajorGemm(handle, false, lhs_gpu, false, rhs_gpu, &output_gpu));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }
  LogBenchmarkArguments(state);
}

BENCHMARK(BM_CublasRowMajorGemm)->Apply(GemmBenchmarkArgs);

void BM_CublasRowMajorGemmTN(benchmark::State& state) {
  BenchmarkUseRealTime();
  int m = state.range(0);
  int k = state.range(1);
  int n = state.range(2);

  // Create the lhs, rhs, and output matrices on gpu.
  absl::BitGen generator;
  CudaMatrix<float> lhs_gpu(k, m, &generator);
  CudaMatrix<float> rhs_gpu(k, n, &generator);
  CudaMatrix<float> output_gpu(m, n, &generator);

  // Setup CuBLAS specific data structures.
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      // Run the cublas kernel.
      CUBLAS_CALL(
          RowMajorGemm(handle, true, lhs_gpu, false, rhs_gpu, &output_gpu));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }
  LogBenchmarkArguments(state);
}

BENCHMARK(BM_CublasRowMajorGemmTN)->Apply(GemmBenchmarkArgs);

}  // namespace sputnik
