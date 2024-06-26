// BSD License

// For "ExecuTorch" software

// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright 2023 Arm Limited and/or its affiliates.
// Copyright (c) Qualcomm Innovation Center, Inc.
// Copyright (c) 2023 Apple Inc.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

//  * Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

//  * Neither the name Meta nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <iostream>
#include <memory>
#include <gflags/gflags.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <opencv2/opencv.hpp>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(image_path, "image.png", "Image path");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

int main(int argc, char** argv) {
  runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const char* model_path = FLAGS_model_path.c_str();

  // Load Program.
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Load Method.
  const char* method_name = "forward";
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans; 
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  MemoryManager memory_manager(&method_allocator, &planned_memory);


  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  // Prepare inputs.
  float data[1 * 1 * 28 * 28] = {0};
  Tensor::SizesType sizes[] = {1, 1, 28, 28};
  const char* image_path = FLAGS_image_path.c_str();
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      data[i * 28 + j] =
          (static_cast<float>(img.at<uchar>(i, j)) - 127.5f) / 127.5f;
    }
  }
  TensorImpl impl(ScalarType::Float, 4, sizes, data);
  Tensor t(&impl);
  Error set_input_error = method->set_input(t, 0);
  ET_LOG(Info, "Inputs prepared.");

  // Run the model.
  Error status = method->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)status);
  ET_LOG(Info, "Model executed successfully.");

  // Print the outputs.
  std::vector<EValue> outputs(1);
  status = method->get_outputs(outputs.data(), 1);
  ET_CHECK(status == Error::Ok);
  std::cout << torch::executor::util::evalue_edge_items(5);
  std::cout << "Output: " << outputs[0] << std::endl;
  return 0;
}
