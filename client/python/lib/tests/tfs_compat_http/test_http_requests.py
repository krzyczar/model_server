#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest


from ovmsclient.tfs_compat.http.requests import (HttpPredictRequest, make_predict_request)

from tfs_compat_http.config import (MODEL_SPEC_INVALID,
                                    PREDICT_REQUEST_INVALID_INPUTS,
                                    PREDICT_REQUEST_VALID)


@pytest.mark.parametrize("inputs, expected_parsed_inputs, name, version", PREDICT_REQUEST_VALID)
def test_make_predict_request_valid(inputs, expected_parsed_inputs, name, version):
    model_predict_request = make_predict_request(inputs, name, version)

    parsed_inputs = model_predict_request.parsed_inputs

    assert isinstance(model_predict_request, HttpPredictRequest)
    assert model_predict_request.model_name == name
    assert model_predict_request.model_version == version
    assert model_predict_request.inputs == inputs
    assert isinstance(parsed_inputs, str)
    assert parsed_inputs == expected_parsed_inputs


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_predict_request_invalid_model_spec(mocker, name, version,
                                                 expected_exception, expected_message):
    inputs = {
        "input": [1, 2, 3]
    }
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.causes_deprecation_warning
@pytest.mark.parametrize("""inputs, name, version,
                            expected_exception, expected_message""", PREDICT_REQUEST_INVALID_INPUTS)
def test_make_predict_request_invalid_inputs(mocker, inputs, name, version,
                                             expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.http.requests._check_model_spec')
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()
