# BSD License

# For "ExecuTorch" software

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2023 Arm Limited and/or its affiliates.
# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright (c) 2023 Apple Inc.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


cmake_minimum_required(VERSION 3.23...3.29)
project(build_model VERSION 0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/executorch")

set(CMAKE_BUILD_TYPE "Debug")
set(EXECUTORCH_ENABLE_LOGGING 1)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(
  "${EXECUTORCH_ROOT}"
  "${CMAKE_BINARY_DIR}/executorch"
)

add_executable(model_runner main.cpp)
target_sources(model_runner PRIVATE "${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp")
target_link_libraries(
  model_runner
  PRIVATE executorch
          gflags
          extension_module_static
          extension_data_loader
          extension_runner_util
          optimized_native_cpu_ops_lib
          xnnpack_backend
          ${OpenCV_LIBRARIES}
)
