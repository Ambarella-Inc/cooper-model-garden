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

import os
import time
from pathlib import Path

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torchvision.transforms.functional import InterpolationMode

import onnx
from onnxsim import simplify
import torch.nn.utils.prune as prune


def export_to_onnx(model : torch.nn.Module, input_tensor : torch.Tensor, output : str, opset : int = 16):
    tmp = 'tmp.onnx'
    # Call the export API. Keep batch as dynamic for batched ONNX evaluation.
    torch.onnx.export(
        model,
        input_tensor,
        tmp,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
                'images': {
                    0: 'batch'
                },
                'output': {
                    0: 'batch'
                }
            }
        )

    # Load the model, simplify using ONNX sim and save the simplified model
    model_onnx = onnx.load(tmp)
    model_onnx, _ = simplify(model_onnx)
    onnx.save(model_onnx, output)
    os.remove(tmp)

def main(args):
    print(args)
    device = torch.device(args.device)

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=args.num_classes)
    model.to(device).eval()
    sample_input_tensor = torch.rand(1,3,args.image_size,args.image_size).type(torch.FloatTensor).to('cuda')

    # Add PyTorch pruning API
    if args.is_sparse:
        print("Apply masks for loading sparse checkpoints")
        prune.global_unstructured(
            parameters=[(module, 'weight') for module in model.modules() if hasattr(module, 'weight')],
            pruning_method=prune.L1Unstructured,
            amount=0.0
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # Call the export to ONNX method
    print("Exporting")
    export_to_onnx(
            model=model,
            input_tensor=sample_input_tensor,
            output=args.output
        )

    print("Exported model: {}".format(args.output))


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--output", default="model.onnx", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument('--image-size', default=224, type=int, help='image size')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes of the dataset the model is trained for')
    parser.add_argument("--is-sparse", action="store_true", help="Apply prune APIs if the model saved is a sparse model")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
