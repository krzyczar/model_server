//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "test_utils.hpp"

#include <functional>

#include "../prediction_service_utils.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

void preparePredictRequest(::inference::ModelInferRequest& request, inputs_info_kfs_t requestInputs, const std::vector<float>& data) {
 //   ::inference::ModelInferRequest request;
    request.mutable_inputs()->Clear();
    request.mutable_raw_input_contents()->Clear();
    for (auto const& it : requestInputs) {
        prepareKFSInferInputTensor(request, it.first, it.second, data);
    }
}

void preparePredictRequest(tensorflow::serving::PredictRequest& request, inputs_info_t requestInputs, const std::vector<float>& data) {
    //    tensorflow::serving::PredictRequest request;
    request.mutable_inputs()->clear();
    for (auto const& it : requestInputs) {
        auto& name = it.first;
        auto [shape, dtype] = it.second;

        auto& input = (*request.mutable_inputs())[name];
        input.set_dtype(dtype);
        size_t numberOfElements = 1;
        for (auto const& dim : shape) {
            input.mutable_tensor_shape()->add_dim()->set_size(dim);
            numberOfElements *= dim;
        }
        switch (dtype) {
        case tensorflow::DataType::DT_HALF: {
            if (data.size() == 0) {
                for (size_t i = 0; i < numberOfElements; i++) {
                    input.add_half_val('1');
                }
            } else {
                for (size_t i = 0; i < data.size(); i++) {
                    input.add_half_val(data[i]);
                }
            }
            break;
        }
        case tensorflow::DataType::DT_UINT16: {
            if (data.size() == 0) {
                for (size_t i = 0; i < numberOfElements; i++) {
                    input.add_int_val('1');
                }
            } else {
                for (size_t i = 0; i < data.size(); i++) {
                    input.add_int_val(data[i]);
                }
            }
            break;
        }
        default: {
            if (data.size() == 0) {
                // TODO in case of DT_HALF & DT_UINT16 we add tensor content two times
                *input.mutable_tensor_content() = std::string(numberOfElements * tensorflow::DataTypeSize(dtype), '1');
            } else {
                std::string content;
                content.resize(numberOfElements * tensorflow::DataTypeSize(dtype));
                std::memcpy(content.data(), data.data(), content.size());
                *input.mutable_tensor_content() = content;
            }
        }
        }
    }
}

void waitForOVMSConfigReload(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.2 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.2;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalSec() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

void waitForOVMSResourcesCleanup(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.2 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.2;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getResourcesCleanupIntervalSec() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

std::string createConfigFileWithContent(const std::string& content, std::string filename) {
    std::ofstream configFile{filename};
    spdlog::info("Creating config file: {}\n with content:\n{}", filename, content);
    configFile << content << std::endl;
    configFile.close();
    if (configFile.fail()) {
        spdlog::info("Closing configFile failed");
    } else {
        spdlog::info("Closing configFile succeed");
    }
    return filename;
}

ovms::tensor_map_t prepareTensors(
    const std::unordered_map<std::string, ovms::Shape>&& tensors,
    ovms::Precision precision) {
    ovms::tensor_map_t result;
    for (const auto& kv : tensors) {
        result[kv.first] = std::make_shared<ovms::TensorInfo>(
            kv.first,
            precision,
            kv.second);
    }
    return result;
}

void checkDummyResponse(const std::string outputName,
    const std::vector<float>& requestData,
    PredictRequest& request, PredictResponse& response, int seriesLength, int batchSize) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_content().size(), batchSize * DUMMY_MODEL_OUTPUT_SIZE * sizeof(float));
    ASSERT_EQ(output_proto.tensor_shape().dim_size(), 2);
    ASSERT_EQ(output_proto.tensor_shape().dim(0).size(), batchSize);
    ASSERT_EQ(output_proto.tensor_shape().dim(1).size(), DUMMY_MODEL_OUTPUT_SIZE);

    std::vector<float> responseData = requestData;
    std::for_each(responseData.begin(), responseData.end(), [seriesLength](float& v) { v += 1.0 * seriesLength; });

    float* actual_output = (float*)output_proto.tensor_content().data();
    float* expected_output = responseData.data();
    const int dataLengthToCheck = DUMMY_MODEL_OUTPUT_SIZE * batchSize * sizeof(float);
    EXPECT_EQ(0, std::memcmp(actual_output, expected_output, dataLengthToCheck))
        << readableError(expected_output, actual_output, dataLengthToCheck / sizeof(float));
}

void checkIncrement4DimShape(const std::string outputName,
    PredictResponse& response,
    const std::vector<size_t>& expectedShape) {
    ASSERT_EQ(response.outputs().count(outputName), 1) << "Did not find:" << outputName;
    const auto& output_proto = response.outputs().at(outputName);

    ASSERT_EQ(output_proto.tensor_shape().dim_size(), expectedShape.size());
    for (size_t i = 0; i < expectedShape.size(); i++) {
        ASSERT_EQ(output_proto.tensor_shape().dim(i).size(), expectedShape[i]);
    }
}

bool isShapeTheSame(const tensorflow::TensorShapeProto& actual, const std::vector<int64_t>&& expected) {
    bool same = true;
    if (static_cast<unsigned int>(actual.dim_size()) != expected.size()) {
        SPDLOG_ERROR("Unexpected dim_size. Got: {}, Expect: {}", actual.dim_size(), expected.size());
        return false;
    }
    for (int i = 0; i < actual.dim_size(); i++) {
        if (actual.dim(i).size() != expected[i]) {
            SPDLOG_ERROR("Unexpected dim[{}]. Got: {}, Expect: {}", i, actual.dim(i).size(), expected[i]);
            same = false;
        }
    }
    if (same == false) {
        std::stringstream ss;
        for (int i = 0; i < actual.dim_size(); i++) {
            ss << "dim["
               << i
               << "] got:"
               << actual.dim(i).size()
               << " expect:" << expected[i];
        }
        SPDLOG_ERROR("Shape mismatch: {}", ss.str());
    }
    return same;
}

extern bool isShapeTheSame(const google::protobuf::RepeatedField<int64_t>& actual, const std::vector<int64_t>&& expected) {
    bool same = true;
    int a_size = actual.size();
    if (a_size != int(expected.size())) {
        SPDLOG_ERROR("Unexpected dim_size. Got: {}, Expect: {}", a_size, expected.size());
        return false;
    }
    for (int i = 0; i < a_size; i++) {
        if (actual.at(i) != expected[i]) {
            SPDLOG_ERROR("Unexpected dim[{}]. Got: {}, Expect: {}", i, actual.at(i), expected[i]);
            same = false;
            break;
        }
    }
    if (same == false) {
        std::stringstream ss;
        for (int i = 0; i < a_size; i++) {
            ss << "dim["
               << i
               << "] got:"
               << actual.at(i)
               << " expect:" << expected[i];
        }
        SPDLOG_ERROR("Shape mismatch: {}", ss.str());
    }
    return same;
}

void readImage(const std::string& path, size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    std::ifstream DataFile;
    DataFile.open(path, std::ios::binary);
    DataFile.seekg(0, std::ios::end);
    filesize = DataFile.tellg();
    DataFile.seekg(0);
    image_bytes = std::make_unique<char[]>(filesize);
    DataFile.read(image_bytes.get(), filesize);
}

void readRgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readImage("/ovms/src/test/binaryutils/rgb.jpg", filesize, image_bytes);
}

void read4x4RgbJpg(size_t& filesize, std::unique_ptr<char[]>& image_bytes) {
    return readImage("/ovms/src/test/binaryutils/rgb4x4.jpg", filesize, image_bytes);
}

tensorflow::serving::PredictRequest prepareBinaryPredictRequest(const std::string& inputName, const int batchSize) {
    tensorflow::serving::PredictRequest request;
    auto& tensor = (*request.mutable_inputs())[inputName];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    readRgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor.add_string_val(image_bytes.get(), filesize);
    }
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
    tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    return request;
}

tensorflow::serving::PredictRequest prepareBinary4x4PredictRequest(const std::string& inputName, const int batchSize) {
    tensorflow::serving::PredictRequest request;
    auto& tensor = (*request.mutable_inputs())[inputName];
    size_t filesize = 0;
    std::unique_ptr<char[]> image_bytes = nullptr;
    read4x4RgbJpg(filesize, image_bytes);

    for (int i = 0; i < batchSize; i++) {
        tensor.add_string_val(image_bytes.get(), filesize);
    }
    tensor.set_dtype(tensorflow::DataType::DT_STRING);
    tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    return request;
}

::inference::ModelInferRequest_InferInputTensor* findKFSInferInputTensor(::inference::ModelInferRequest& request, const std::string& name) {
    auto it = request.mutable_inputs()->begin();
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
    }
    return it == request.mutable_inputs()->end() ? nullptr : &(*it);
}

std::string* findKFSInferInputTensorContent(::inference::ModelInferRequest& request, const std::string& name) {
    auto it = request.mutable_inputs()->begin();
    size_t bufferId = 0;
    std::string* content = nullptr;
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
        ++bufferId;
    }
    if (it != request.mutable_inputs()->end()) {
        content = request.mutable_raw_input_contents()->Mutable(bufferId);
    }
    return content;
}

void prepareKFSInferInputTensor(::inference::ModelInferRequest& request, const std::string& name, const std::tuple<ovms::shape_t, const std::string>& inputInfo,
    const std::vector<float>& data) {
    auto it = request.mutable_inputs()->begin();
    size_t bufferId = 0;
    while (it != request.mutable_inputs()->end()) {
        if (it->name() == name)
            break;
        ++it;
        ++bufferId;
    }
    ::inference::ModelInferRequest_InferInputTensor* tensor;
    std::string* content;
    if (it != request.mutable_inputs()->end()) {
        tensor = &*it;
        content = request.mutable_raw_input_contents()->Mutable(bufferId);
    } else {
        tensor = request.add_inputs();
        content = request.add_raw_input_contents();
    }
    auto [shape, dtype] = inputInfo;
    tensor->set_name(name);
    tensor->set_datatype(dtype);
    size_t numberOfElements = 1;
    tensor->mutable_shape()->Clear();
    for (auto const& dim : shape) {
        tensor->add_shape(dim);
        numberOfElements *= dim;
    }
    if (data.size() == 0) {
        content->assign(numberOfElements * ovms::KFSDataTypeSize(dtype), '1');
    } else {
        content->resize(numberOfElements * ovms::KFSDataTypeSize(dtype));
        std::memcpy(content->data(), data.data(), content->size());
    }
}
