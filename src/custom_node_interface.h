//*****************************************************************************
// Copyright 2021 Intel Corporation
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
//*****************************************************************************
#pragma once

#include <stdint.h>

typedef enum {
    UNSPECIFIED,
    FP32,
    FP16,
    U8,
    I8,
    I16,
    U16,
    I32
} CustomNodeTensorPrecision;

struct CustomNodeTensor {
    const char* name;
    uint8_t* data;
    uint64_t dataBytes;
    uint64_t* dims;
    uint64_t dimsCount;
    CustomNodeTensorPrecision precision;
};

struct CustomNodeTensorInfo {
    const char* name;
    uint64_t* dims;
    uint64_t dimsCount;
    CustomNodeTensorPrecision precision;
};

struct CustomNodeParam {
    const char *key, *value;
};

#ifdef __cplusplus
extern "C" {
#endif

// custom node library initialize for optimized buffers allocation
// using initialize is optional and not required for custom node to work
// customNodeLibraryInternalManager should be created here if initialize is used
// on initialize failure approperiate status is returned and error log is printed
int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
// custom node library deinitialize for optimized buffers allocation
// using deinitialize is optional and not required for custom node to work
// customNodeLibraryInternalManager should be destroyed here if deinitialize is used
// on deinitialize failure only log error is printed
int deinitialize(void* customNodeLibraryInternalManager);
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int release(void* ptr, void* customNodeLibraryInternalManager);

#ifdef __cplusplus
}
#endif
