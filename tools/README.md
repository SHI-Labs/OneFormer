# OneFormer Tools

## Download Pretrained Weights

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. We use [Swin-Tranformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), and [DiNAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) for our experiments.

<details>
<summary>Swin-Transformer</summary>

- [Official Repo](https://github.com/microsoft/Swin-Transformer)
- `convert-pretrained-model-to-d2.py`: Tool to convert Swin Transformer pre-trained weights for D2.

    ```bash
    pip install timm

    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    python tools/convert-pretrained-model-to-d2.py swin_large_patch4_window12_384_22k.pth swin_large_patch4_window12_384_22k.pkl

    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth
    python tools/convert-pretrained-model-to-d2.py swin_large_patch4_window12_384_22kto1k.pth swin_large_patch4_window12_384_22kto1k.pkl
    ```

</details>

<details>
<summary>ConvNeXt</summary>

- [Official Repo](https://github.com/facebookresearch/ConvNeXt)
- `convert-pretrained-model-to-d2.py`: Tool to convert ConvNeXt pre-trained weights for D2.

    ```bash
    wget https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth
    python tools/convert-pretrained-model-to-d2.py convnext_large_22k_1k_384.pth convnext_large_22k_1k_384.pkl

    wget https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth
    python tools/convert-pretrained-model-to-d2.py convnext_xlarge_22k_1k_384_ema.pth convnext_xlarge_22k_1k_384_ema.pkl
    ```

</details>

<details>
<summary>DiNAT</summary>

- [Official Repo](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
- `convert-pretrained-nat-model-to-d2.py`: Tool to convert DiNAT pre-trained weights for D2.

    ```bash
    wget https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_large_in22k_in1k_384_11x11.pth
    python tools/convert-pretrained-nat-model-to-d2.py dinat_large_in22k_in1k_384_11x11.pth dinat_large_in22k_in1k_384_11x11.pkl

    wget https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth
    python tools/convert-pretrained-nat-model-to-d2.py dinat_large_in22k_224.pth dinat_large_in22k_224.pkl
    ```
    
</details>

## Analyze Model

- Tool to analyze model parameters, flops and speed.
- We use dummy image to compute flops on ADE20K and Cityscapes.
- For COCO, we use random 100 validation images.
- We set `task = panoptic` by default.

```bash
python tools/analyze_model.py --num-inputs 100 --tasks [flop speed] \
    --config-file configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml \
    MODEL.WEIGHTS <path-to-checkpoint> [--use-fixed-input-size] MODEL.TEST.SEMANTIC_ON False MODEL.TEST.INSTANCE_ON False
```

## Training Throughput

- Tool to compute throughput.
- We compute throughput for 500 iterations by default.

```bash
python tools/calc_throughput.py --dist-url 'tcp://127.0.0.1:50162' \
--num-gpus 8 \
--config-file configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml \
MODEL.WEIGHTS pretrain/swin_large_patch4_window12_384_22kto1k.pkl \
OUTPUT_DIR tp_out SOLVER.MAX_ITER 500

rm -rf tp_out
```
