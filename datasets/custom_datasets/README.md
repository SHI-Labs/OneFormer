# Training OneFormer with Custom Datasets

OneFormer advocates the usage of panoptic annotations along with its task-conditioned joint training strategy. However, if panoptic annotations are not available, then also OneFormer can be trained using only the instance or semantic annotations on custom datasets. We provide some guidelines for training with custom datasets.

## Register your New Dataset

- OneFormer uses the information (class names, thing classes, etc.) stored in a dataset's metadata while preparing a dataset dictionary using a [`dataset_mapper`](https://github.com/SHI-Labs/OneFormer/tree/main/oneformer/data/dataset_mappers).

- [Use Custom Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) gives a deeper dive into registering a new custom dataset.

## Training with Available Panoptic Annotations

- To prepare the dataset dictionary for each iteration during training, OneFormer uses a [`dataset_mapper`](https://github.com/SHI-Labs/OneFormer/tree/main/oneformer/data/dataset_mappers) class.

- Originally, we provide two `dataset_mapper` classes which support task-conditioned joint training using the panoptic annotations:  
  - [`COCOUnifiedNewBaselineDatasetMapper`](https://github.com/SHI-Labs/OneFormer/blob/5e04c9aaffd9bc73020d2238757f62346fe778c0/oneformer/data/dataset_mappers/coco_unified_new_baseline_dataset_mapper.py#L56): Specifically designed for COCO annotation format.
  - [`OneFormerUnifiedDatasetMapper`](https://github.com/SHI-Labs/OneFormer/blob/5e04c9aaffd9bc73020d2238757f62346fe778c0/oneformer/data/dataset_mappers/oneformer_unified_dataset_mapper.py#L26): General annotation format.

- If you have panoptic annotations for your custom dataset, you may use these dataset_mapper classes directly after registering your dataset. You may also tune the [task sampling probabilities in the corresponding config file](https://github.com/SHI-Labs/OneFormer/blob/5e04c9aaffd9bc73020d2238757f62346fe778c0/configs/ade20k/Base-ADE20K-UnifiedSegmentation.yaml#L55).

- If you want to train using only the instance or semantic annotation, please follow the next section on preparing a custom dataset mapper class.

## Write a Custom Dataset Mapper Class

- If you want to train using only instance or semantic annotations, write your custom dataset mapper class and add it to the [`build_train_loader`](https://github.com/SHI-Labs/OneFormer/blob/5e04c9aaffd9bc73020d2238757f62346fe778c0/train_net.py#L156) method.

- We provide a few templates for custom dataset mappers:
  - [`InstanceCOCOCustomNewBaselineDatasetMapper`](https://github.com/SHI-Labs/OneFormer/blob/a7fae86ce5791a93132c059c1bdfc79c9f842820/datasets/custom_datasets/instance_coco_custom_dataset_mapper.py#L72): Specifically designed for COCO instance annotation format.
  - [`InstanceOneFormerCustomDatasetMapper`](https://github.com/SHI-Labs/OneFormer/blob/a7fae86ce5791a93132c059c1bdfc79c9f842820/datasets/custom_datasets/instance_oneformer_custom_dataset_mapper.py#L26): General instance annotation format.
  - [`SemanticOneFormerCustomDatasetMapper`](https://github.com/SHI-Labs/OneFormer/blob/a7fae86ce5791a93132c059c1bdfc79c9f842820/datasets/custom_datasets/semantic_oneformer_custom_dataset_mapper.py#L26): General semantic annotation format.

- Remember to register your custom dataset before training. 


Now you are all set to train OneFormer using your custom dataset!