This document records information for LLaVA-OneVision Qwen2 7B model.

## LLaVA-NeXT Git Repository

The LLaVA-NeXT git repository is needed to load the model and export the vision tower and mm projector models to ONNX.

The following is the repository status.

```
commit b3a46be22d5aa818fa1a23542ae3a28f8e2ed421 (origin/main, origin/HEAD)
Merge: 4f2a2fe eb55586
Author: Li Bo <drluodian@gmail.com>
Date:   Wed Sep 25 10:47:02 2024 +0800

    Merge pull request #250 from litianjian/main

    Fix typos
```

## LLaVA-OneVision HuggingFace model

The LLaVA-OneVision model refers to the one located on HuggingFace.

https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov-chat
```
commit 2f1f5be47db85478a45722c28e76bc4f314eaaa5 (HEAD -> main, origin/main, origin/HEAD)
Author: Bo Li <luodian@users.noreply.huggingface.co>
Date:   Fri Sep 13 15:52:45 2024 +0000

    Update README.md
```

## Export

Please use `export_onnx.py` to get the ONNX model.