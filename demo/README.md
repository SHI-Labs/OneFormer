# OneFormer Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/shi-labs/OneFormer)

- Pick a model and its config file from. For example, `configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml`.
- We provide `demo.py` that is able to demo builtin configs.
- You need to specify the `task` token value during inference, The outputs will be saved accordingly in the specified `OUTPUT_DIR`:
  - `panoptic`: Panoptic, Semantic and Instance Predictions when the value of `task` token is `panoptic`.
  - `instance`: Instance Predictions when the value of `task` token is `instance`.
  - `semantic`: Semantic Predictions when the value of `task` token is `semantic`.
  - >Note: You can change the outputs to be saved on line 60 in [predictor.py](predictor.py).

```bash
export task=panoptic

python demo.py --config-file ../configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
  --input <path-to-images> \
  --output <output-path> \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS <path-to-checkpoint>
```

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. 