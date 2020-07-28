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

#include <cstdint>

#include "glog/logging.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/dlmc.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace sputnik {

// Identifiers for models - useful to passing model argument to
// benchmarks that only accept integer arguments.
enum Model {
  TRANSFORMER = 0,
  RESNET_50 = 1,
  EFFICIENTNET_B1 = 2,
};

void BenchmarkArgs(testing::Benchmark* b) {
  const int kNumTransformerMatrices = 2716;
  for (int idx = 0; idx < kNumTransformerMatrices; ++idx) {
    b->Args({TRANSFORMER, idx, 256});
    b->Args({TRANSFORMER, idx, 2048});
  }
  const int kNumResNetMatrices = 1130;
  for (int idx = 0; idx < kNumResNetMatrices; ++idx) {
    // Load the batch size from the file and scale by 1 for
    // inference batch size. Training batch size would be
    // scaled by 256.
    b->Args({RESNET_50, idx, 1});
    b->Args({RESNET_50, idx, 256});
  }
  const int kNumEfficientNetMatrices = 45;
  for (int idx = 0; idx < kNumEfficientNetMatrices; ++idx) {
    // Load the batch size from the file and scale by 1 for
    // inference batch size. Training batch size would be
    // scaled by 256.
    b->Args({EFFICIENTNET_B1, idx, 1});
  }
}

std::string GetFileName(int model_id, int matrix_id) {
  std::string filename;
  if (model_id == TRANSFORMER) {
    filename = dlmc::TransformerFiles().at(matrix_id);
  } else if (model_id == RESNET_50) {
    filename = dlmc::ResNetFiles().at(matrix_id);
  } else if (model_id == EFFICIENTNET_B1) {
    filename = dlmc::EfficientNetFiles().at(matrix_id);
  } else {
    LOG(ERROR) << "Model ID " << model_id << "exceeds valid range.";
  }
  return filename;
}

SparseMatrix* LoadMatrix(int model_id, int matrix_id, absl::BitGen* generator,
                         Swizzle row_swizzle, int row_padding) {
  std::string filename = GetFileName(model_id, matrix_id);
  int rows, columns, nonzeros;
  std::vector<int> row_offsets, column_indices;
  dlmc::LoadMatrix(filename, &rows, &columns, &nonzeros, &row_offsets,
                   &column_indices);

  // Move the matrix into a SparseMatrix object.
  SparseMatrix* out =
      new SparseMatrix(rows, columns, nonzeros, row_offsets, column_indices,
                       generator, row_swizzle, row_padding);
  return out;
}

int GetBatchSize(benchmark::State& state) {
  if (state.range(0) == RESNET_50) {
    const auto& rn50_filenames = dlmc::ResNetFiles();
    return dlmc::ResNetBatchSize(rn50_filenames[state.range(1)]) *
           state.range(2);
  } else if (state.range(0) == EFFICIENTNET_B1) {
    const auto& enb1_filenames = dlmc::EfficientNetFiles();
    return dlmc::EfficientNetBatchSize(enb1_filenames[state.range(1)]) *
           state.range(2);
  }
  return state.range(2);
}

void BM_CudaSddmm_GenericFloat(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // If the batch size is not divisible by 4, pad so that we can still use
  // 4-wide vector instructions. This is only needed for some ResNet-50
  // inference problems.
  int batch_size = (kBatchSize + 3) / 4 * 4;

  // No padding required for correctness.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<float> sparse_matrix(*matrix_staging);
  delete matrix_staging;

  // Create the lhs and rhs matrices on the gpu.
  CudaMatrix<float> lhs_matrix(sparse_matrix.Rows(), batch_size, &generator);
  CudaMatrix<float> rhs_matrix(sparse_matrix.Columns(), batch_size, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << kBatchSize << ","
            << "n=" << sparse_matrix.Columns() << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      CUDA_CALL(CudaSddmm(
          sparse_matrix.Rows(), lhs_matrix.Columns(), sparse_matrix.Columns(),
          sparse_matrix.NumElementsWithPadding(), sparse_matrix.RowIndices(),
          sparse_matrix.RowOffsets(), sparse_matrix.ColumnIndices(),
          lhs_matrix.Values(), rhs_matrix.Values(), sparse_matrix.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
}

BENCHMARK(BM_CudaSddmm_GenericFloat)->Apply(BenchmarkArgs);

void BM_CusparseSddmm_ConstrainedGemm(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // No row padding needed.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<float> sparse_matrix(*matrix_staging);
  delete matrix_staging;

  // Create the lhs and rhs matrices on the gpu.
  CudaMatrix<float> lhs_matrix(sparse_matrix.Rows(), kBatchSize, &generator);
  CudaMatrix<float> rhs_matrix(kBatchSize, sparse_matrix.Columns(), &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << kBatchSize << ","
            << "n=" << sparse_matrix.Columns() << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;

  // Create the lhs matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_lhs_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(
      &cusparse_lhs_matrix, lhs_matrix.Rows(), lhs_matrix.Columns(),
      lhs_matrix.Rows(), lhs_matrix.Values(), CUDA_R_32F, CUSPARSE_ORDER_COL));

  // Create the rhs matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_rhs_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(
      &cusparse_rhs_matrix, rhs_matrix.Rows(), rhs_matrix.Columns(),
      rhs_matrix.Rows(), rhs_matrix.Values(), CUDA_R_32F, CUSPARSE_ORDER_COL));

  // Create the sparse matrix in cusparse format.
  cusparseSpMatDescr_t cusparse_sparse_matrix;
  CUSPARSE_CALL(cusparseCreateCsr(
      &cusparse_sparse_matrix, sparse_matrix.Rows(), sparse_matrix.Columns(),
      sparse_matrix.NumElementsWithPadding(), sparse_matrix.RowOffsets(),
      sparse_matrix.ColumnIndices(), sparse_matrix.Values(), CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // Allocate scratchpad memory for the computation.
  char* scratchpad = nullptr;
  size_t buffer_size = 0;
  CUSPARSE_CALL(cusparseConstrainedGeMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cusparse_lhs_matrix,
      cusparse_rhs_matrix, &beta, cusparse_sparse_matrix, CUDA_R_32F,
      &buffer_size));
  CUDA_CALL(cudaMalloc(&scratchpad, buffer_size));

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      // NOTE: CuSparse currently only supports non-tranposed operands for
      // SDDMM. Both input dense matrices are taken to be column-major,
      // and the output matrix is stored in compressed sparse row. We
      // currently setup the right-hand matrix s.t. it is already transposed
      // and don't include the transposition in the benchmark.
      CUSPARSE_CALL(cusparseConstrainedGeMM(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cusparse_lhs_matrix,
          cusparse_rhs_matrix, &beta, cusparse_sparse_matrix, CUDA_R_32F,
          scratchpad));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // Cleanup.
  CUDA_CALL(cudaFree(scratchpad));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_lhs_matrix));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_rhs_matrix));
  CUSPARSE_CALL(cusparseDestroySpMat(cusparse_sparse_matrix));
}

BENCHMARK(BM_CusparseSddmm_ConstrainedGemm)->Apply(BenchmarkArgs);

void BM_CusparseSddmm_ConstrainedGemt(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // No row padding needed.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<float> sparse_matrix(*matrix_staging);
  delete matrix_staging;

  // Create the lhs and rhs matrices on the gpu.
  CudaMatrix<float> lhs_matrix(sparse_matrix.Rows(), kBatchSize, &generator);
  CudaMatrix<float> rhs_matrix(sparse_matrix.Columns(), kBatchSize, &generator);

  // Buffer for the transposed rhs matrix.
  CudaMatrix<float> rhs_matrix_t(kBatchSize, sparse_matrix.Columns(),
                                 &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << kBatchSize << ","
            << "n=" << sparse_matrix.Columns() << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;

  // Create the lhs matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_lhs_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(
      &cusparse_lhs_matrix, lhs_matrix.Rows(), lhs_matrix.Columns(),
      lhs_matrix.Rows(), lhs_matrix.Values(), CUDA_R_32F, CUSPARSE_ORDER_COL));

  // Create the (transposed) rhs matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_rhs_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(&cusparse_rhs_matrix, rhs_matrix_t.Rows(),
                                    rhs_matrix_t.Columns(), rhs_matrix_t.Rows(),
                                    rhs_matrix_t.Values(), CUDA_R_32F,
                                    CUSPARSE_ORDER_COL));

  // Create the sparse matrix in cusparse format.
  cusparseSpMatDescr_t cusparse_sparse_matrix;
  CUSPARSE_CALL(cusparseCreateCsr(
      &cusparse_sparse_matrix, sparse_matrix.Rows(), sparse_matrix.Columns(),
      sparse_matrix.NumElementsWithPadding(), sparse_matrix.RowOffsets(),
      sparse_matrix.ColumnIndices(), sparse_matrix.Values(), CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // Allocate scratchpad memory for the computation.
  char* scratchpad = nullptr;
  size_t buffer_size = 0;
  CUSPARSE_CALL(cusparseConstrainedGeMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cusparse_lhs_matrix,
      cusparse_rhs_matrix, &beta, cusparse_sparse_matrix, CUDA_R_32F,
      &buffer_size));
  CUDA_CALL(cudaMalloc(&scratchpad, buffer_size));

  // Setup CuBLAS specific data structures.
  cublasHandle_t cublas_handle;
  CUBLAS_CALL(cublasCreate(&cublas_handle));
  CUBLAS_CALL(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

  int batch_iterations = 1000;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      // NOTE: CuSparse currently only supports non-tranposed operands for
      // SDDMM. Do the explicit transpose before calling the sddmm.
      CUBLAS_CALL(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              rhs_matrix.Columns(), rhs_matrix.Rows(), &alpha,
                              rhs_matrix.Values(), rhs_matrix.Rows(), &beta,
                              rhs_matrix.Values(),   // unused
                              rhs_matrix.Columns(),  // unused
                              rhs_matrix_t.Values(), rhs_matrix.Columns()));

      CUSPARSE_CALL(cusparseConstrainedGeMM(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cusparse_lhs_matrix,
          cusparse_rhs_matrix, &beta, cusparse_sparse_matrix, CUDA_R_32F,
          scratchpad));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // Cleanup.
  CUDA_CALL(cudaFree(scratchpad));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_lhs_matrix));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_rhs_matrix));
  CUSPARSE_CALL(cusparseDestroySpMat(cusparse_sparse_matrix));
}

BENCHMARK(BM_CusparseSddmm_ConstrainedGemt)->Apply(BenchmarkArgs);

}  // namespace sputnik
