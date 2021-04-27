#!/bin/bash
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

# Variables setup
export OVMS_PATH=/opt/model_server
export KALDI_PATH=/opt/kaldi
export ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
export DATA_PATH=/opt/data

# Extract features 
source $OVMS_PATH/.venv/bin/activate
cd $OVMS_PATH/example_client/stateful
./asr_demo/prepare_model_inputs.sh
python grpc_stateful_client.py --input_path /opt/data/feats.ark,/opt/data/ivectors.ark --output_path /opt/data/scores.ark --grpc_address localhost --grpc_port 9000 --input_name input,ivector --output_name Final_affine --model_name aspire --cw_l 17 --cw_r 12
./asr_demo/read_model_output.sh


