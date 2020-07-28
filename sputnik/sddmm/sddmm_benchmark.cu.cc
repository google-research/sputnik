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

#include "glog/logging.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"
#include "benchmark/benchmark.h"
#include "absl/random/random.h"

namespace sputnik {

void BenchmarkArgs(testing::Benchmark* b) {
  const std::vector<int> kHiddenSizes = {2048, 4096, 8192};
  const std::vector<int> kBatchSizes = {8, 16, 32, 64, 128, 256};
  const std::vector<float> kDensities = {.5f, .375f, .25f, .125f, 0.015625f};
  for (const auto& hs : kHiddenSizes) {
    for (const auto& density : kDensities) {
      for (const auto& bs : kBatchSizes) {
        // Clifford I2H
        b->Args({hs * 4, bs, hs,
                 static_cast<int>(std::round(hs * hs * 4 * density))});

        // Clifford HM
        b->Args({hs, bs, hs, static_cast<int>(std::round(hs * hs * density))});

        // Clifford H2H
        b->Args({3 * hs, bs, hs,
                 static_cast<int>(std::round(hs * hs * 3 * density))});
      }
    }
  }
}

void LogBenchmarkArguments(const benchmark::State& state) {
  // Print the arguments to the benchmark.
  LOG(INFO) << "Benchmark Arguments: "
            << "m=" << state.range(0) << ","
            << "k=" << state.range(1) << ","
            << "n=" << state.range(2) << ","
            << "nonzeros=" << state.range(3);
}

typedef std::function<cudaError_t(int, int, int, int, const int*, const int*,
                                  const int*, const float*, const float*,
                                  float*, cudaStream_t)>
    SddmmFn;

void BenchmarkFn(SddmmFn sddmm_fn, benchmark::State& state) {
  BenchmarkUseRealTime();
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);

  // No padding required for correctness.
  const int kRowPadding = 0;

  // Create the sparse matrix on the gpu.
  absl::BitGen generator;
  CudaSparseMatrix<float> output_matrix(kDimM, kDimN, kNonZeros, RANDOM_UNIFORM,
                                        &generator, SORTED, kRowPadding);

  // Create the dense matrix on the gpu.
  CudaMatrix<float> lhs_matrix(kDimM, kDimK, &generator);
  CudaMatrix<float> rhs_matrix(kDimN, kDimK, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(sddmm_fn(
          output_matrix.Rows(), lhs_matrix.Columns(), output_matrix.Columns(),
          output_matrix.NumElementsWithPadding(), output_matrix.RowIndices(),
          output_matrix.RowOffsets(), output_matrix.ColumnIndices(),
          lhs_matrix.Values(), rhs_matrix.Values(), output_matrix.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(nullptr));
  }

  // log the benchmark arguments for parsing
  LogBenchmarkArguments(state);
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ANONYMOUS_NAME(x) CONCAT(x, __COUNTER__)

#define REGISTER_BENCHMARK(name, fn)                                  \
  void BM_##name(benchmark::State& state) { BenchmarkFn(fn, state); } \
  BENCHMARK(BM_##name)->Apply(BenchmarkArgs)

#define REGISTER_TILED_BENCHMARK_HELPER(name, tname, fn, ltype, mt, kt, nt, \
                                        bs)                                 \
  const auto& tname = fn<ltype, mt, kt, nt, bs>;                            \
  REGISTER_BENCHMARK(name##_##ltype##x##mt##x##kt##x##nt##x##bs, tname)

#define REGISTER_TILED_BENCHMARK(name, fn, ltype, mt, kt, nt, bs)            \
  REGISTER_TILED_BENCHMARK_HELPER(name, ANONYMOUS_NAME(sddmm_fn), fn, ltype, \
                                  mt, kt, nt, bs)

REGISTER_TILED_BENCHMARK(CudaSddmmEx, CudaSddmmEx, float, 1, 32, 32, 32);
REGISTER_TILED_BENCHMARK(CudaSddmmEx, CudaSddmmEx, float2, 2, 32, 32, 16);
REGISTER_TILED_BENCHMARK(CudaSddmmEx, CudaSddmmEx, float4, 4, 32, 32, 8);

}  // namespace sputnik
