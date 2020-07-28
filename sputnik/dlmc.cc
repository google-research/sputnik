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

#include "sputnik/dlmc.h"

#include <unordered_map>

#include "glog/logging.h"
#include "file/base/filelineiter.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace sputnik {
namespace dlmc {

namespace {

const char kDlmcBasePath[] = "dlmc/";

std::vector<std::string> ReadFilenames(std::string base_path,
                                       std::string filename) {
  // Read the matrix paths in line by line and save for easy access.
  FileLines filelines(base_path + filename, FileLineIterator::REMOVE_LINEFEED,
                      /* max_line_length = */ 1e3);
  std::vector<std::string> output;
  for (const auto &line : filelines) {
    output.push_back(base_path + line);
  }
  return output;
}

std::vector<std::string> MakeTransformerFilenames() {
  return ReadFilenames(std::string(kDlmcBasePath), "transformer_matrices.txt");
}

std::vector<std::string> MakeResNetFilenames() {
  return ReadFilenames(std::string(kDlmcBasePath), "rn50_matrices.txt");
}

std::vector<std::string> MakeEfficientNetFilenames() {
  return ReadFilenames(std::string(kDlmcBasePath), "enb1_matrices.txt");
}

std::vector<std::string> MakeMbv1Filenames() {
  return ReadFilenames(std::string(kDlmcBasePath), "mbv1_matrices.txt");
}

std::unordered_map<std::string, int> ReadBatchSizes(std::string filename) {
  FileLines filelines(filename);
  std::unordered_map<std::string, int> output;
  for (const auto &line : filelines) {
    std::vector<std::string> name_and_batch_size =
        absl::StrSplit(line, absl::ByChar(','));
    CHECK_EQ(name_and_batch_size.size(), 2)
        << "Expected line with two entries.";
    CHECK(output.find(name_and_batch_size[0]) == output.end())
        << "Entry for " << name_and_batch_size[0] << "already exists.";

    int batch_size;
    CHECK(absl::SimpleAtoi(name_and_batch_size[1], &batch_size))
        << "Could not convert " << name_and_batch_size[1] << " to int.";
    // Create the mapping.
    output[name_and_batch_size[0]] = batch_size;
  }
  return output;
}

std::unordered_map<std::string, int> MakeResNetBatchSizes() {
  return ReadBatchSizes(std::string(kDlmcBasePath) + "rn50_batchsizes.txt");
}

std::unordered_map<std::string, int> MakeEfficientNetBatchSizes() {
  return ReadBatchSizes(std::string(kDlmcBasePath) + "enb1_batchsizes.txt");
}

std::unordered_map<std::string, int> MakeMbv1BatchSizes() {
  return ReadBatchSizes(std::string(kDlmcBasePath) + "mbv1_batchsizes.txt");
}

// Helper to extract matrix name from dlmc path.
std::string Basename(const std::string &filename) {
  std::vector<std::string> ext_split =
      absl::StrSplit(filename, absl::ByChar('.'));
  CHECK_GE(ext_split.size(), 2)
      << "Expected at least one '.' delimiter in file name '" << filename
      << "'";
  std::vector<std::string> path_split =
      absl::StrSplit(ext_split[ext_split.size() - 2], absl::ByChar('/'));
  CHECK_GT(path_split.size(), 1)
      << "Expected at least one '/' delimiter in file name '" << filename
      << "'";
  return path_split.back();
}

void ExtractValues(const std::string &line, int n, std::vector<int> *values,
                   std::string delimiter = " ") {
  std::vector<std::string> split_line = absl::StrSplit(line, delimiter);
  CHECK_GE(split_line.size(), n)
      << "Expected at least " << n << " delimited values. Found "
      << split_line.size();
  int out;
  for (int i = 0; i < n; ++i) {
    CHECK(absl::SimpleAtoi(split_line[i], &out))
        << "Failed to convert '" << split_line[i] << " to integer.";
    (*values)[i] = out;
  }
}

void ExtractMatrixMetaData(const std::string &line, int *m, int *k, int *nnz) {
  std::vector<int> values(3);
  ExtractValues(line, 3, &values, ", ");
  *m = values[0];
  *k = values[1];
  *nnz = values[2];
}

}  // namespace

const std::vector<std::string> &TransformerFiles() {
  static const std::vector<std::string> kTransformerFiles =
      MakeTransformerFilenames();
  return kTransformerFiles;
}

const std::vector<std::string> &ResNetFiles() {
  static const std::vector<std::string> kResNetFiles = MakeResNetFilenames();
  return kResNetFiles;
}

int ResNetBatchSize(const std::string &filename) {
  static const std::unordered_map<std::string, int> kResNetBatchSizes =
      MakeResNetBatchSizes();
  return kResNetBatchSizes.at(Basename(filename));
}

const std::vector<std::string> &EfficientNetFiles() {
  static const std::vector<std::string> kEfficientNetFiles =
      MakeEfficientNetFilenames();
  return kEfficientNetFiles;
}

int EfficientNetBatchSize(const std::string &filename) {
  static const std::unordered_map<std::string, int> kEfficientNetBatchSizes =
      MakeEfficientNetBatchSizes();
  return kEfficientNetBatchSizes.at(Basename(filename));
}

const std::vector<std::string> &Mbv1Files() {
  static const std::vector<std::string> kMbv1Files = MakeMbv1Filenames();
  return kMbv1Files;
}

int Mbv1BatchSize(const std::string &filename) {
  static const std::unordered_map<std::string, int> kMbv1BatchSizes =
      MakeMbv1BatchSizes();
  return kMbv1BatchSizes.at(Basename(filename));
}

void LoadMatrix(const std::string &filename, int *m, int *k, int *nnz,
                std::vector<int> *row_offsets,
                std::vector<int> *column_indices) {
  FileLines filelines(filename, FileLineIterator::REMOVE_LINEFEED,
                      /* max_line_length = */ 1e9);
  auto lineiter = filelines.begin();

  // Read the matrix meta-data.
  ExtractMatrixMetaData(*lineiter, m, k, nnz);

  // Validate the inputs.
  CHECK_GT(*m, 0) << "Number of rows (" << *m << ") to be greater than zero.";
  CHECK_GT(*k, 0) << "Number of columns (" << *k
                  << ") must be greater than zero.";
  CHECK_GE(*nnz, 0) << "Number of nonzeros (" << *nnz
                    << ") must be greater than or equal to zero.";
  CHECK_LE(*nnz, (*m) * (*k))
      << "Number of nonzeros (" << *nnz << ") must be less than m*k ("
      << (*m) * (*k) << ").";

  // Resize the row indices and column indices buffers.
  row_offsets->resize(*m + 1);
  column_indices->resize(*nnz);

  // Extract the row offsets from the file.
  ++lineiter;
  CHECK(!lineiter.eof())
      << "Encountered end-of-file prior to reading row offsets.";
  ExtractValues(*lineiter, *m + 1, row_offsets);

  ++lineiter;
  CHECK(!lineiter.eof())
      << "Encountered end-of-file prior to reading column indices.";
  ExtractValues(*lineiter, *nnz, column_indices);

  ++lineiter;
  CHECK(lineiter.eof())
      << "Completed reading matrix but did not reach end-of-file.";
}

}  // namespace dlmc
}  // namespace sputnik
