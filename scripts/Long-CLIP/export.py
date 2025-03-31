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
import torch
import argparse
from torch import nn
import numpy as np
from PIL import Image
from model import longclip


class ClipDecoder(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = 100 * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image

device = "cpu"

def load_args():
	parser = argparse.ArgumentParser(
		description="Export the LongCLIP onnx model."
	)
	parser.add_argument(
		"-p", "--pretrained", type=str, required=False, default="./checkpoints/longclip-B.pt",
		help="The path of the LongCLIP pretrained pth model."
	)
	parser.add_argument(
		"-i", "--image_path", type=str, required=False, default="img/framework.PNG",
		help="The image path."
	)
	parser.add_argument(
		"-t", "--text", type=str, required=False, default="a photo of diagram",
		help="The text content that needs to be matched."
	)
	parser.add_argument(
		"-ep", "--export_path", type=str, required=False, default="./onnx_models/longclip-B16/",
		help="The LongCLIP onnx model export path."
	)
	parser.add_argument(
		"-a", "--arch", type=str, required=False, default="b16",
		help="The LongCLIP weights type."
	)

	args = parser.parse_args()

	return args

args = load_args()

os.system("mkdir -p {}/".format(args.export_path))

model, preprocess = longclip.load(args.pretrained, device=device)

image = preprocess(Image.open(args.image_path)).unsqueeze(0).to(device)
text = args.text.split(";")
text = longclip.tokenize(text).to(device)

decoder = ClipDecoder()

with torch.no_grad():
    image_features = model.encode_image(image)

    torch.onnx.export(model.visual,
                    image,
                    "{}/longclip_vit{}_image.onnx".format(args.export_path, args.arch),
                    opset_version = 14,
                    input_names=("images", ),
                    output_names=("image_features", )
                    )

    text_features = model.encode_text(text)

    torch.onnx.export(model,
                    (text,),
                    "{}/longclip_vit{}_text.onnx".format(args.export_path, args.arch),
                    opset_version = 14,
                    input_names=("tokens", ),
                    output_names=("text_features", )
                    )

    logits_per_image = decoder(image_features, text_features)

    probs = logits_per_image.softmax(1).cpu().numpy()

print("Label probs:", probs)   # prints: [[0.9927937  0.00421068 0.00299572]]

os.system("onnxsim {}/longclip_vit{}_image.onnx {}/longclip_vit{}_image_sim.onnx".format(args.export_path, args.arch, args.export_path, args.arch))
os.system("onnxsim {}/longclip_vit{}_text.onnx {}/longclip_vit{}_text_sim.onnx".format(args.export_path, args.arch, args.export_path, args.arch))
