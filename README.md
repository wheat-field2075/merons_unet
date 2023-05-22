### 2023/05/22: dihedral4
- added augmentations for dihedral-4
- loss on val-data is noticably lower, but fails to yield noticable improvments on the entire patch

### 2023/03/17: negative samples and dataset_process
- should be dated as 2023/05/17
- dataset_preprocess.ipynb now generates patches and stores them on disk, rather than creating patches at runtime and storing in memory. should help w/ scalability.
- added negative samples to training, which are defined as patches w/o a patch center in the mask center. patches are randonly sampled from the images and checked.
