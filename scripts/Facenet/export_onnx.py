##################################################################
#	BSD 3-Clause License
#
#	Copyright (c) 2025 Ambarella International LP
#	All rights reserved.
#
#	Redistribution and use in source and binary forms, with or without
#	modification, are permitted provided that the following conditions are met:
#
#	* Redistributions of source code must retain the above copyright notice, this
#	  list of conditions and the following disclaimer.
#
#	* Redistributions in binary form must reproduce the above copyright notice,
#	  this list of conditions and the following disclaimer in the documentation
#	  and/or other materials provided with the distribution.
#
#	* Neither the name of the copyright holder nor the names of its
#	  contributors may be used to endorse or promote products derived from
#	  this software without specific prior written permission.
#
#	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################################

import torch
import nets.facenet as models
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Facenet Model Exporter")
    parser.add_argument('--model-path', type=str, default='model_data/facenet_mobilenet.pth', help='Path to the Facenet model file')
    parser.add_argument('--output-path', type=str, default='./facenet_mobilenet_org.onnx', help='Output path for the ONNX model')
    return parser.parse_args()

def main():
    args = parse_args()
    model=models.Facenet(mode='export')
    device = torch.device('cpu')
    model.load_state_dict(torch.load(args.model_path ,map_location=device),strict=False)
    net=model.eval()
    example=torch.rand(1,3,160,160)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,(example),args.output_path ,input_names=input_names,output_names=output_names,verbose=True,opset_version=13)

if __name__ == "__main__":
    main()