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

#ifndef THIRD_PARTY_SPUTNIK_DLMC_H_
#define THIRD_PARTY_SPUTNIK_DLMC_H_

#include <string>
#include <vector>

namespace sputnik {
namespace dlmc {

/**
 * @brief Returns a list of all transformer weight matrices from dlmc.
 */
const std::vector<std::string> &TransformerFiles();

/**
 * @brief Returns a list of all resnet-50 weight matrices from dlmc.
 */
const std::vector<std::string> &ResNetFiles();

/**
 * @brief Returns the batch size for the given weight matrix name.
 */
int ResNetBatchSize(const std::string &filename);

/**
 * @brief Returns a list of all efficientnet-b1 weight matrices from dlmc.
 */
const std::vector<std::string> &EfficientNetFiles();

/**
 * @brief Returns the batch size for the given weight matrix name.
 */
int EfficientNetBatchSize(const std::string &filename);

/**
 * @brief Returns a list of all mbv1 weight matrices from dlmc.
 */
const std::vector<std::string> &Mbv1Files();

/**
 * @brief Returns the batch size for the given weight matrix name.
 */
int Mbv1BatchSize(const std::string &filename);

/**
 * @brief Loads the specified weight matrix into vectors.
 */
void LoadMatrix(const std::string &filename, int *m, int *k, int *nnz,
                std::vector<int> *row_offsets,
                std::vector<int> *column_indices);

}  // namespace dlmc
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DLMC_H_
