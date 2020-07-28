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
#include "sputnik/spmm/cuda_spmm.h"
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

void BM_CudaSpmm_GenericFloat(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // If the batch size is not divisible by 4, pad so that we can still use
  // 4-wide vector instructions. This is only needed for some ResNet-50
  // inference problems.
  int batch_size = (kBatchSize + 3) / 4 * 4;

  // With the addition of the reverse offset memory alignment trick we
  // no longer need to pad rows of the sparse matrix to use vector memory
  // instruction.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<float> sparse_matrix(*matrix_staging);
  delete matrix_staging;

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix(sparse_matrix.Columns(), batch_size, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix(sparse_matrix.Rows(), batch_size, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << sparse_matrix.Columns() << ","
            << "n=" << kBatchSize << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Safety check for index calculations:
  int64_t max_idx = static_cast<int64_t>(sparse_matrix.Columns() - 1) *
                    batch_size * sizeof(float);
  CHECK_LT(max_idx, INT_MAX)
      << "Pre-calculated pointer offsets can exceed numeric limits.";

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      CUDA_CALL(
          CudaSpmm(sparse_matrix.Rows(), sparse_matrix.Columns(),
                   matrix.Columns(), sparse_matrix.NumElementsWithPadding(),
                   sparse_matrix.RowIndices(), sparse_matrix.Values(),
                   sparse_matrix.RowOffsets(), sparse_matrix.ColumnIndices(),
                   matrix.Values(), output_matrix.Values(), nullptr));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
}

BENCHMARK(BM_CudaSpmm_GenericFloat)->Apply(BenchmarkArgs);

void BM_CudaSpmm_GenericHalf(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // If the batch size is not divisible by 4, pad so that we can still use
  // 4-wide vector instructions. This is only needed for some ResNet-50
  // inference problems.
  int batch_size = (kBatchSize + 7) / 8 * 8;

  // Pad to the nearest 2 nonzero elements so we can use the `half2`
  // datatype.
  const int kRowPadding = 2;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<half2> sparse_matrix(*matrix_staging);
  delete matrix_staging;

  // Create the dense matrix on the gpu.
  CudaMatrix<half2> matrix(sparse_matrix.Columns(), batch_size, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<half2> output_matrix(sparse_matrix.Rows(), batch_size, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << sparse_matrix.Columns() << ","
            << "n=" << kBatchSize << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      CUDA_CALL(
          CudaSpmm(sparse_matrix.Rows(), sparse_matrix.Columns(),
                   matrix.Columns(), sparse_matrix.NumElementsWithPadding(),
                   sparse_matrix.RowIndices(), sparse_matrix.Values(),
                   sparse_matrix.RowOffsets(), sparse_matrix.ColumnIndices(),
                   matrix.Values(), output_matrix.Values(), nullptr));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
}

BENCHMARK(BM_CudaSpmm_GenericHalf)->Apply(BenchmarkArgs);

void BM_CusparseSpmm_Csrmm2(benchmark::State& state) {
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

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix(sparse_matrix.Columns(), kBatchSize, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix(sparse_matrix.Rows(), kBatchSize, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << sparse_matrix.Columns() << ","
            << "n=" << kBatchSize << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;
  cusparseMatDescr_t mdesc;
  CUSPARSE_CALL(cusparseCreateMatDescr(&mdesc));
  CUSPARSE_CALL(cusparseSetMatIndexBase(mdesc, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(cusparseSetMatType(mdesc, CUSPARSE_MATRIX_TYPE_GENERAL));

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
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
          CUSPARSE_OPERATION_TRANSPOSE, sparse_matrix.Rows(), kBatchSize,
          sparse_matrix.Columns(), sparse_matrix.NumElementsWithPadding(),
          &alpha, mdesc, sparse_matrix.Values(), sparse_matrix.RowOffsets(),
          sparse_matrix.ColumnIndices(), matrix.Values(), kBatchSize, &beta,
          output_matrix.Values(), sparse_matrix.Rows()));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
}

BENCHMARK(BM_CusparseSpmm_Csrmm2)->Apply(BenchmarkArgs);

void BM_CusparseSpmm_Csrmm(benchmark::State& state) {
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

  // Create the dense matrix on the gpu.
  CudaMatrix<float> matrix(sparse_matrix.Columns(), kBatchSize, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<float> output_matrix(sparse_matrix.Rows(), kBatchSize, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << sparse_matrix.Columns() << ","
            << "n=" << kBatchSize << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;
  cusparseMatDescr_t mdesc;
  CUSPARSE_CALL(cusparseCreateMatDescr(&mdesc));
  CUSPARSE_CALL(cusparseSetMatIndexBase(mdesc, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(cusparseSetMatType(mdesc, CUSPARSE_MATRIX_TYPE_GENERAL));

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      // CuSparse SpMM variant with both right-hand size and output matrices
      // stored in column-major format.
      CUSPARSE_CALL(cusparseScsrmm(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE, sparse_matrix.Rows(),
          kBatchSize, sparse_matrix.Columns(),
          sparse_matrix.NumElementsWithPadding(), &alpha, mdesc,
          sparse_matrix.Values(), sparse_matrix.RowOffsets(),
          sparse_matrix.ColumnIndices(), matrix.Values(), matrix.Rows(), &beta,
          output_matrix.Values(), output_matrix.Rows()));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }
}

BENCHMARK(BM_CusparseSpmm_Csrmm)->Apply(BenchmarkArgs);

void BM_CusparseSpmm_SpmmHalf(benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kModelId = state.range(0);
  const int kMatrixId = state.range(1);
  const int kBatchSize = GetBatchSize(state);

  // Pad to the nearest 2 nonzero elements so we can use the `half2`
  // datatype.
  const int kRowPadding = 2;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  SparseMatrix* matrix_staging =
      LoadMatrix(kModelId, kMatrixId, &generator, SORTED, kRowPadding);
  CudaSparseMatrix<float> sparse_matrix(*matrix_staging);
  CudaSparseMatrix<half2> sparse_matrix_fp16(*matrix_staging);
  delete matrix_staging;

  // Create the dense matrix on the gpu.
  CudaMatrix<half2> matrix(sparse_matrix.Columns(), kBatchSize, &generator);

  // Create the output matrix on the gpu.
  CudaMatrix<half2> output_matrix(sparse_matrix.Rows(), kBatchSize, &generator);

  // Log the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "fname=" << GetFileName(kModelId, kMatrixId) << ","
            << "m=" << sparse_matrix.Rows() << ","
            << "k=" << sparse_matrix.Columns() << ","
            << "n=" << kBatchSize << ","
            << "nonzeros=" << sparse_matrix.Nonzeros() << ","
            << "model_id=" << kModelId << ","
            << "matrix_id=" << kMatrixId;
  if (sparse_matrix.Nonzeros() == 0) return;

  // Setup CuSparse specific data structures.
  cusparseHandle_t handle;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  float alpha = 1.0, beta = 0.0;

  // Create the sparse matrix in cusparse format. Note that
  // CuSparse only supports 32-bit indices.
  cusparseSpMatDescr_t cusparse_sparse_matrix;
  CUSPARSE_CALL(cusparseCreateCsr(
      &cusparse_sparse_matrix, sparse_matrix.Rows(), sparse_matrix.Columns(),
      sparse_matrix.NumElementsWithPadding(), sparse_matrix.RowOffsets(),
      sparse_matrix.ColumnIndices(), sparse_matrix_fp16.Values(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_16F));

  // Create the rhs matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_rhs_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(
      &cusparse_rhs_matrix, matrix.Rows(), matrix.Columns(), matrix.Rows(),
      matrix.Values(), CUDA_R_16F, CUSPARSE_ORDER_COL));

  // Create the output matrix in cusparse format.
  cusparseDnMatDescr_t cusparse_output_matrix;
  CUSPARSE_CALL(cusparseCreateDnMat(
      &cusparse_output_matrix, output_matrix.Rows(), output_matrix.Columns(),
      output_matrix.Rows(), output_matrix.Values(), CUDA_R_16F,
      CUSPARSE_ORDER_COL));

  // Allocate scratchpad memory for the computation.
  char* scratchpad = nullptr;
  size_t buffer_size = 0;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cusparse_sparse_matrix,
      cusparse_rhs_matrix, &beta, cusparse_output_matrix, CUDA_R_32F,
      CUSPARSE_CSRMM_ALG1, &buffer_size));
  CUDA_CALL(cudaMalloc(&scratchpad, buffer_size));

  int batch_iterations = 10;
  while (state.KeepRunningBatch(batch_iterations)) {
    for (int i = 0; i < batch_iterations; ++i) {
      CUSPARSE_CALL(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                 cusparse_sparse_matrix, cusparse_rhs_matrix,
                                 &beta, cusparse_output_matrix, CUDA_R_32F,
                                 CUSPARSE_CSRMM_ALG1, scratchpad));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // Cleanup.
  CUDA_CALL(cudaFree(scratchpad));
  CUSPARSE_CALL(cusparseDestroySpMat(cusparse_sparse_matrix));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_rhs_matrix));
  CUSPARSE_CALL(cusparseDestroyDnMat(cusparse_output_matrix));
}

BENCHMARK(BM_CusparseSpmm_SpmmHalf)->Apply(BenchmarkArgs);

}  // namespace sputnik
