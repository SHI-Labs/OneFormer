# Getting Started with OneFormer

This document provides a brief intro of the usage of OneFormer.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

## Training

- Make sure to setup wandb before training a model.

  ```bash
  pip install wandb
  wandb login
  ```

- We provide a script `train_net.py`, that is made to train all the configs provided in OneFormer.

- To train a model with "train_net.py", first setup the corresponding datasets following [datasets/README.md](./datasets/README.md).

- Be default, the model uses `task=panoptic` for evaluation during training.

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 8 \
    --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
    OUTPUT_DIR outputs/ade20k_swin_large WANDB.NAME ade20k_swin_large
```

## Evaluation

- You need to pass the value of `task` token. `task` belongs to [panoptic, semantic, instance].

- To evaluate a model's performance, use:

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 8 \
    --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>
```

## Inference Demo

We provide a demo script for inference on images. For more information, please see [demo/README.md](demo/README.md).
