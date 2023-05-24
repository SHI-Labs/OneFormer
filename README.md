# OneFormer: One Transformer to Rule Universal Image Segmentation

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb) [![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shi-labs/OneFormer) [![HuggingFace transformers](https://img.shields.io/badge/🤗-HuggingFace%20transformers-magenta.svg)](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/_Zr1pOi7Chw) [![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/instance-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/instance-segmentation-on-cityscapes-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/instance-segmentation-on-coco-val-panoptic)](https://paperswithcode.com/sota/instance-segmentation-on-coco-val-panoptic?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/panoptic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-ade20k-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/instance-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/instance-segmentation-on-ade20k-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/panoptic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-cityscapes-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/panoptic-segmentation-on-cityscapes-test)](https://paperswithcode.com/sota/panoptic-segmentation-on-cityscapes-test?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/panoptic-segmentation-on-mapillary-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-mapillary-val?p=oneformer-one-transformer-to-rule-universal) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/semantic-segmentation-on-mapillary-val)](https://paperswithcode.com/sota/semantic-segmentation-on-mapillary-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/semantic-segmentation-on-coco-1)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-1?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=oneformer-one-transformer-to-rule-universal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer-one-transformer-to-rule-universal/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=oneformer-one-transformer-to-rule-universal)

[Jitesh Jain](https://praeclarumjj3.github.io/), [Jiachen Li](https://chrisjuniorli.github.io/)<sup>&dagger;</sup>, [MangTik Chiu](https://www.linkedin.com/in/mtchiu/)<sup>&dagger;</sup>, [Ali Hassani](https://alihassanijr.com/), [Nikita Orlov](https://www.linkedin.com/in/nukich74/), [Humphrey Shi](https://www.humphreyshi.com/home)

<sup>&dagger;</sup> Equal Contribution

[[`Project Page`](https://praeclarumjj3.github.io/oneformer/)] [[`arXiv`](https://arxiv.org/abs/2211.06220)] [[`pdf`](https://arxiv.org/pdf/2211.06220.pdf)] [[`BibTeX`](#4citation)]

This repo contains the code for our paper **OneFormer: One Transformer to Rule Universal Image Segmentation**.

<img src="images/teaser.png" width="100%"/>

#### Features

- OneFormer is the **first** multi-task universal image segmentation framework based on transformers.
- OneFormer needs to be trained only once with a single universal architecture, a single model, and on a single dataset , to outperform existing frameworks across semantic, instance, and panoptic segmentation tasks.
- OneFormer uses a task-conditioned joint training strategy, uniformly sampling different ground truth domains (semantic instance, or panoptic) by deriving all labels from panoptic annotations to train its multi-task model.
- OneFormer uses a task token to condition the model on the task in focus, making our architecture task-guided for training, and task-dynamic for inference, all with a single model.

![OneFormer](images/oneformer.svg)

## Contents

1. [News](#news)
2. [Installation Instructions](#installation-instructions)
3. [Dataset Preparation](#dataset-preparation)
4. [Execution Instructions](#execution-instructions)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Demo](#demo)
5. [Results](#results)
6. [Citation](#citation)

## News

- **[February 27, 2023]**: OneFormer is accepted to CVPR 2023!
- **[January 26, 2023]**: OneFormer sets new SOTA performance on the the Mapillary Vistas val (both panoptic & semantic segmentation) and Cityscapes test (panoptic segmentation) sets. We’ve released the checkpoints too!
- **[January 19, 2023]**: OneFormer is now available as a part of the 🤗 **HuggingFace [transformers](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) library** and **[model hub](https://huggingface.co/models?filter=oneformer)**! 🚀
- **[December 26, 2022]**: Checkpoints for Swin-L OneFormer and DiNAT-L OneFormer trained on ADE20K with 1280&times;1280 resolution released!
- **[November 23, 2022]**: Roboflow cover OneFormer on [YouTube](https://youtu.be/_Zr1pOi7Chw)! Thanks to [@SkalskiP](https://github.com/SkalskiP) for making the video!
- **[November 18, 2022]**: Our demo is available on 🤗 [Huggingface Space](https://huggingface.co/spaces/shi-labs/OneFormer)!
- **[November 10, 2022]**: [**Project Page**](https://praeclarumjj3.github.io/oneformer/), [**ArXiv Preprint**](https://praeclarumjj3.github.io/oneformer/) and [**GitHub Repo**](https://praeclarumjj3.github.io/oneformer/) are public!
  - OneFormer sets new SOTA on Cityscapes val with single-scale inference on Panoptic Segmentation with **68.5** PQ score and Instance Segmentation with **46.7** AP score!
  - OneFormer sets new SOTA on ADE20K val on Panoptic Segmentation with **51.5** PQ score and on Instance Segmentation with **37.8** AP!
  - OneFormer sets new SOTA on COCO val on Panoptic Segmentation with **58.0** PQ score!

## Installation Instructions

- We use Python 3.8, PyTorch 1.10.1 (CUDA 11.3 build).
- We use Detectron2-v0.6.
- For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## Dataset Preparation

- We experiment on three major benchmark dataset: ADE20K, Cityscapes and COCO 2017.
- Please see [Preparing Datasets for OneFormer](datasets/README.md) for complete instructions for preparing the datasets.

## Execution Instructions

### Training

- We train all our models using 8 A6000 (48 GB each) GPUs.
- We use 8 A100 (80 GB each) for training Swin-L<sup>&dagger;</sup> OneFormer and DiNAT-L<sup>&dagger;</sup> OneFormer on COCO and all models with ConvNeXt-XL<sup>&dagger;</sup> backbone. We also train the 896x896 models on ADE20K on 8 A100 GPUs.
- Please see [Getting Started with OneFormer](GETTING_STARTED.md) for training commands.

### Evaluation

- Please see [Getting Started with OneFormer](GETTING_STARTED.md) for evaluation commands.

### Demo

- We provide quick to run demos on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb) and Hugging Face Spaces [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/shi-labs/OneFormer).
- Please see [OneFormer Demo](demo/README.md) for command line instructions on running the demo.

## Results

![Results](images/plots.svg)

- &dagger; denotes the backbones were pretrained on ImageNet-22k.
- Pre-trained models can be downloaded following the instructions given [under tools](tools/README.md/#download-pretrained-weights).

### ADE20K

| Method | Backbone | Crop Size |  PQ   | AP   | mIoU <br> (s.s) | mIoU <br> (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    |  :---:    | :---: | :---:| :---:           | :---:               | :---:   |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 640&times;640 | 49.8 | 35.9 | 57.0 | 57.7 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | Swin-L<sup>&dagger;</sup> | 896&times;896 | 51.1 | 37.6 | 57.4 | 58.3 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | Swin-L<sup>&dagger;</sup> | 1280&times;1280 | 51.4 | 37.8 | 57.0 | 57.7 | 219M | [config](configs/ade20k/swin/oneformer_swin_large_bs16_160k_1280x1280.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_swin_l_oneformer_ade20k_160k.pth) |
| OneFormer | ConvNeXt-L<sup>&dagger;</sup> | 640&times;640 | 50.0 | 36.2 | 56.6 | 57.4 | 220M | [config](configs/ade20k/convnext/oneformer_convnext_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 640&times;640 | 50.5 | 36.0 | 58.3 | 58.4 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 896&times;896 | 51.2 | 36.8 | 58.1 | 58.6 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_896x896.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 1280&times;1280 | 51.5 | 37.1 | 58.3 | 58.7 | 223M | [config](configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth) |
| OneFormer (COCO-Pretrained) | DiNAT-L<sup>&dagger;</sup> | 1280&times;1280 | 53.4 | 40.2 | 58.4 | 58.8 | 223M | [config](configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280_coco_pretrain.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) |
| OneFormer | ConvNeXt-XL<sup>&dagger;</sup> | 640&times;640 | 50.1 | 36.3 | 57.4 | 58.8 | 372M | [config](configs/ade20k/convnext/oneformer_convnext_xlarge_bs16_160k.yaml) | [model](https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_xl_oneformer_ade20k_160k.pth) |

### Cityscapes

| Method | Backbone |  PQ   | AP   | mIoU <br> (s.s) | mIoU <br> (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---: | :---:| :---:      | :---:          | :---:   |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 67.2 | 45.6 | 83.0 | 84.4 | 219M | [config](configs/cityscapes/swin/oneformer_swin_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth) |
| OneFormer | ConvNeXt-L<sup>&dagger;</sup> | 68.5 | 46.5 | 83.0 | 84.0 | 220M | [config](configs/cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_l_oneformer_cityscapes_90k.pth) |
| OneFormer (Mapillary Vistas-Pretrained) | ConvNeXt-L<sup>&dagger;</sup> | 70.1 | 48.7 | 84.6 | 85.2 | 220M | [config](configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_l_oneformer_cityscapes_90k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/mapillary/mapillary_pretrain_250_16_convnext_l_oneformer_mapillary_300k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 67.6 | 45.6 | 83.1 | 84.0 | 223M | [config](configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth) |
| OneFormer | ConvNeXt-XL<sup>&dagger;</sup> | 68.4 | 46.7 | 83.6 | 84.6 | 372M | [config](configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_xl_oneformer_cityscapes_90k.pth) |
| OneFormer (Mapillary Vistas-Pretrained) | ConvNeXt-XL<sup>&dagger;</sup> | 69.7 | 48.9 | 84.5 | 85.8 | 372M | [config](configs/cityscapes/convnext/mapillary_pretrain_oneformer_convnext_xlarge_bs16_90k.yaml) | [model](https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth) &#124; [pretrained](https://shi-labs.com/projects/oneformer/mapillary/mapillary_pretrain_250_16_convnext_xl_oneformer_mapillary_300k.pth) |

### COCO

| Method | Backbone |  PQ   |  PQ<sup>Th</sup>   |  PQ<sup>St</sup>   | AP | mIoU | #params | config | Checkpoint |
|   :---:| :---:    | :---: | :---:              | :---:              |:---:| :---:| :---:  |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 57.9 | 64.4 | 48.0 | 49.0 | 67.4 | 219M | [config](configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 58.0 | 64.3 | 48.4 | 49.2 | 68.1 | 223M | [config](configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) |

### Mapillary Vistas

| Method | Backbone |  PQ   | mIoU <br> (s.s) | mIoU <br> (ms+flip) | #params | config | Checkpoint |
|   :---:| :---:    | :---: |:---:            | :---:               | :---:  |  :---: |    :---:   |
| OneFormer | Swin-L<sup>&dagger;</sup> | 46.7 | 62.9 | 64.1 | 219M | [config](configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml) | [model](https://shi-labs.com/projects/oneformer/mapillary/250_16_swin_l_oneformer_mapillary_300k.pth) |
| OneFormer | ConvNeXt-L<sup>&dagger;</sup> | 47.9 | 63.2 | 63.8 | 220M | [config](configs/mapillary_vistas/convnext/oneformer_convnext_large_bs16_300k.yaml) | [model](https://shi-labs.com/projects/oneformer/mapillary/250_16_convnext_l_oneformer_mapillary_300k.pth) |
| OneFormer | DiNAT-L<sup>&dagger;</sup> | 47.8 | 64.0 | 64.9 | 223M | [config](configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml) | [model](https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth) |


## Citation

If you found OneFormer useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@inproceedings{jain2023oneformer,
      title={{OneFormer: One Transformer to Rule Universal Image Segmentation}},
      author={Jitesh Jain and Jiachen Li and MangTik Chiu and Ali Hassani and Nikita Orlov and Humphrey Shi},
      journal={CVPR}, 
      year={2023}
    }
```

## Acknowledgement

We thank the authors of [Mask2Former](https://github.com/facebookresearch/Mask2Former), [GroupViT](https://github.com/NVlabs/GroupViT), and [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) for releasing their helpful codebases.
