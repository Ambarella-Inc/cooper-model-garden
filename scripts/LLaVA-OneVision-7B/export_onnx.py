##################################################################
#       BSD 3-Clause License
#
#       Copyright (c) 2025 Ambarella International LP
#       All rights reserved.
#
#       Redistribution and use in source and binary forms, with or without
#       modification, are permitted provided that the following conditions are met:
#
#       * Redistributions of source code must retain the above copyright notice, this
#         list of conditions and the following disclaimer.
#
#       * Redistributions in binary form must reproduce the above copyright notice,
#         this list of conditions and the following disclaimer in the documentation
#         and/or other materials provided with the distribution.
#
#       * Neither the name of the copyright holder nor the names of its
#         contributors may be used to endorse or promote products derived from
#         this software without specific prior written permission.
#
#       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#       AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#       IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#       DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#       FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#       DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#       SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#       CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#       OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#       OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################################

import torch
import torch.nn as nn
import sys
import warnings

sys.path.append("LLaVA-NeXT")
warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model

class MultiImage(nn.Module):
    def __init__(self, vision_tower, mm_projector, image_newline):
        super(MultiImage, self).__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.image_newline = image_newline

    def forward(self, image):
        image_features_vt = self.vision_tower(image.half())
        image_features_mmp = self.mm_projector(image_features_vt)
        image_features = torch.cat((image_features_mmp, self.image_newline[None, None].to(image_features_mmp.device)), dim=1)

        return image_features


image = torch.rand(1, 3, 384, 384, dtype=torch.float16).cuda()

pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
        "multimodal": True,
    }
overwrite_config = {}
overwrite_config["image_aspect_ratio"] = "pad"
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()

vision_tower = model.get_model().get_vision_tower()
mm_projector = model.get_model().mm_projector
image_newline = model.model.image_newline

multi_image_model = MultiImage(vision_tower, mm_projector, image_newline)

onnx_fn = "llavaonevision_multi_image_mode_self_contained.onnx"
torch.onnx.export(
    multi_image_model,
    (image, {}),
    onnx_fn,
    input_names=["image"],
    output_names=["image_features"],
    opset_version=16,
    verbose=False,
    keep_initializers_as_inputs=True,
    do_constant_folding=True
)

