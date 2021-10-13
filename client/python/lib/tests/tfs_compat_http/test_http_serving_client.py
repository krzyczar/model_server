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
import requests

from ovmsclient.tfs_compat.http.serving_client import HttpClient, make_http_client
from ovmsclient.tfs_compat.http.responses import (HttpModelStatusResponse,
                                                  HttpModelMetadataResponse,
                                                  HttpPredictResponse)
from ovmsclient.tfs_compat.http.requests import (HttpModelStatusRequest,
                                                 HttpModelMetadataRequest,
                                                 HttpPredictRequest)

from tfs_compat_http.config import (BUILD_VALID, BUILD_INVALID_CONFIG,
                                    MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE,
                                    MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE,
                                    PREDICT_REQUEST_INVALID_REQUEST_TYPE)


@pytest.fixture
def valid_http_serving_client_min():
    return make_http_client("localhost:9000")


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, text):
            self.text = text

    return MockResponse(args)


@pytest.mark.parametrize("config, method_call_count, expected_client_key, expected_server_cert",
                         BUILD_VALID)
def test_make_http_client_valid(mocker, config, method_call_count, expected_client_key,
                                expected_server_cert):
    mock_check_url = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                  '.ServingClient._check_url')
    mock_check_tls_config = mocker.patch('ovmsclient.tfs_compat.base.serving_client'
                                         '.ServingClient._check_tls_config')
    client = make_http_client(**config)

    assert mock_check_url.call_count == method_call_count['_check_url']
    assert mock_check_tls_config.call_count == method_call_count['_check_tls_config']
    assert client.url == config['url']
    assert client.client_key == expected_client_key
    assert client.server_cert == expected_server_cert
    assert isinstance(client.session, requests.Session)


@pytest.mark.parametrize("config, method_call_spec, expected_exception, expected_message",
                         BUILD_INVALID_CONFIG)
def test_make_http_client_invalid_config(mocker, config, method_call_spec, expected_exception,
                                         expected_message):
    mocks = []
    for method_name, call_spec in method_call_spec.items():
        call_count, error_raised = call_spec
        mock = mocker.patch(f"ovmsclient.tfs_compat.base.serving_client." \
                            f"ServingClient.{method_name}", side_effect=error_raised)
        mocks.append((mock, call_count))

    with pytest.raises(expected_exception) as e_info:
        make_http_client(**config)

    assert str(e_info.value) == expected_message
    for mock_info in mocks:
        mock, call_count = mock_info
        assert mock.call_count == call_count


@pytest.mark.parametrize("text", [
    """{
         "model_version_status": [
          {
           "version": "1",
           "state": "AVAILABLE",
           "status": {
            "error_code": "OK",
            "error_message": "OK"
           }
          }
         ]
        }"""
])
def test_get_model_status_valid(mocker, valid_http_serving_client_min,
                                text):
    model_status_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_status_request')

    raw_response = mocked_requests_get(text)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    response = valid_http_serving_client_min.get_model_status(model_status_request)

    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.get.call_count == 1
    assert type(response) == HttpModelStatusResponse
    assert response.raw_response == raw_response
    assert response.raw_response.text == raw_response.text


@pytest.mark.parametrize("expected_message", [
    ("('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"),
    ("HTTPConnectionPool(host='localhost', port=54000): "
     "Max retries exceeded with url: /v1/models/resnet/versions/1 "
     "(Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7ff21a41f370>: "
     "Failed to establish a new connection: [Errno 111] Connection refused'))")
])
def test_get_model_status_invalid_http(mocker, valid_http_serving_client_min,
                                       expected_message):
    model_status_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_status_request')

    valid_http_serving_client_min.session.get\
        = mocker.Mock(side_effect=requests.exceptions.ConnectionError(expected_message))

    with pytest.raises(ConnectionError) as e_info:
        valid_http_serving_client_min.get_model_status(model_status_request)

    assert str(e_info.value) == ("There was an error during sending ModelStatusRequest. "
                                 "Http exited with:\n" + expected_message)
    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.get.call_count == 1


@pytest.mark.parametrize("model_status_request, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE)
def test_get_model_status_invalid_request_type(mocker, valid_http_serving_client_min,
                                               model_status_request, expected_exception,
                                               expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_status_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_http_serving_client_min.get_model_status(model_status_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


def test_check_model_status_request_valid():
    HttpClient._check_model_status_request(HttpModelStatusRequest("model_name", 0))


@pytest.mark.parametrize("model_status_request, expected_exception,"
                         "expected_message", MODEL_STATUS_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_status_request_invalid_type(model_status_request, expected_exception,
                                                 expected_message):
    with pytest.raises(expected_exception) as e_info:
        HttpClient._check_model_status_request(model_status_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("text", [
    """{
        'modelSpec': {
            'name': 'ssd_mobilenet',
            'signatureName': '',
            'version': '1'
            },
        'metadata': {
            'signature_def': {
                '@type': 'type.googleapis.com/tensorflow.serving.SignatureDefMap',
                'signatureDef': {
                    'serving_default': {
                        'inputs': {
                            'image_tensor': {
                                'dtype': 'DT_FLOAT',
                                'tensorShape': {
                                    'dim': [
                                        {'size': '1', 'name': ''},
                                        {'size': '3', 'name': ''},
                                        {'size': '300', 'name': ''},
                                        {'size': '300', 'name': ''}],
                                    'unknownRank': False
                                },
                                'name': 'image_tensor'
                            }
                        },
                        'outputs': {
                            'DetectionOutput': {
                                'dtype': 'DT_FLOAT',
                                'tensorShape': {
                                    'dim': [
                                        {'size': '1', 'name': ''},
                                        {'size': '1', 'name': ''},
                                        {'size': '100', 'name': ''},
                                        {'size': '7', 'name': ''}],
                                    'unknownRank': False
                                },
                                'name': 'DetectionOutput'
                            }
                        },
                    'methodName': ''
                    }
                }
            }
        }
    }
"""
])
def test_get_model_metadata_valid(mocker, valid_http_serving_client_min,
                                  text):
    model_metadata_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_metadata_request')

    raw_response = mocked_requests_get(text)
    valid_http_serving_client_min.session.get\
        = mocker.Mock(return_value=raw_response)

    response = valid_http_serving_client_min.get_model_metadata(model_metadata_request)

    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.get.call_count == 1
    assert type(response) == HttpModelMetadataResponse
    assert response.raw_response == raw_response
    assert response.raw_response.text == raw_response.text


@pytest.mark.parametrize("expected_message", [
    ("('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"),
    ("HTTPConnectionPool(host='localhost', port=54000): "
     "Max retries exceeded with url: /v1/models/resnet/versions/1/metadata "
     "(Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7ff21a41f370>: "
     "Failed to establish a new connection: [Errno 111] Connection refused'))")
])
def test_get_model_metadata_invalid_http(mocker, valid_http_serving_client_min,
                                         expected_message):
    model_metadata_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_metadata_request')

    valid_http_serving_client_min.session.get\
        = mocker.Mock(side_effect=requests.exceptions.ConnectionError(expected_message))

    with pytest.raises(ConnectionError) as e_info:
        valid_http_serving_client_min.get_model_metadata(model_metadata_request)

    assert str(e_info.value) == ("There was an error during sending ModelMetadataRequest. "
                                 "Http exited with:\n" + expected_message)
    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.get.call_count == 1


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_get_model_metadata_invalid_request_type(mocker, valid_http_serving_client_min,
                                                 model_metadata_request, expected_exception,
                                                 expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_model_metadata_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_http_serving_client_min.get_model_metadata(model_metadata_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


def test_check_model_metadata_request_valid():
    HttpClient._check_model_metadata_request(HttpModelMetadataRequest("model_name", 0))


@pytest.mark.parametrize("model_metadata_request, expected_exception,"
                         "expected_message", MODEL_METADATA_REQUEST_INVALID_REQUEST_TYPE)
def test_check_model_metadata_request_invalid_type(model_metadata_request, expected_exception,
                                                   expected_message):
    with pytest.raises(expected_exception) as e_info:
        HttpClient._check_model_metadata_request(model_metadata_request)

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("text", [
    """{
        "outputs": {
            "softmax_tensor": [[0.1, 0.22, 0.01, 0.67]]
        }
    }"""
])
def test_predict_valid(mocker, valid_http_serving_client_min, text):
    predict_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_predict_request')

    raw_response = mocked_requests_get(text)
    valid_http_serving_client_min.session.post\
        = mocker.Mock(return_value=raw_response)

    response = valid_http_serving_client_min.predict(predict_request)

    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.post.call_count == 1
    assert type(response) == HttpPredictResponse
    assert response.raw_response == raw_response
    assert response.raw_response.text == raw_response.text


@pytest.mark.parametrize("expected_message", [
    ("('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"),
    ("HTTPConnectionPool(host='localhost', port=54000): "
     "Max retries exceeded with url: /v1/models/resnet/versions/1:predict "
     "(Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7ff21a41f370>: "
     "Failed to establish a new connection: [Errno 111] Connection refused'))")
])
def test_predict_invalid_http(mocker, valid_http_serving_client_min,
                              expected_message):
    predict_request = mocker.Mock()

    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_predict_request')

    valid_http_serving_client_min.session.post\
        = mocker.Mock(side_effect=requests.exceptions.ConnectionError(expected_message))

    with pytest.raises(ConnectionError) as e_info:
        valid_http_serving_client_min.predict(predict_request)

    assert str(e_info.value) == ("There was an error during sending PredictRequest. "
                                 "Http exited with:\n" + expected_message)
    assert mock_check_request.call_count == 1
    assert valid_http_serving_client_min.session.post.call_count == 1


@pytest.mark.parametrize("predict_request, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_REQUEST_TYPE)
def test_predict_invalid_request_type(mocker, valid_http_serving_client_min,
                                      predict_request, expected_exception,
                                      expected_message):
    mock_check_request = mocker.patch('ovmsclient.tfs_compat.http.serving_client'
                                      '.HttpClient._check_predict_request',
                                      side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        valid_http_serving_client_min.predict(predict_request)

    assert str(e_info.value) == expected_message
    assert mock_check_request.call_count == 1


def test_check_predict_request_valid():
    HttpClient._check_predict_request(HttpPredictRequest({}, "model_name", 0, "{}"))


@pytest.mark.parametrize("predict_request, expected_exception,"
                         "expected_message", PREDICT_REQUEST_INVALID_REQUEST_TYPE)
def test_check_predict_invalid_type(predict_request, expected_exception,
                                    expected_message):
    with pytest.raises(expected_exception) as e_info:
        HttpClient._check_predict_request(predict_request)

    assert str(e_info.value) == expected_message
