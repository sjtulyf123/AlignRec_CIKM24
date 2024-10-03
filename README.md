# AlignRec_CIKM24

This is the official code for [AlignRec: Aligning and Training in Multimodal Recommendations](https://arxiv.org/abs/2403.12384).

## Code Introduction
We develop our repo based on the [MMRec](https://github.com/enoche/MMRec) arch, so please first download all necessary data mentioned in it.

We list the main changes of AlignRec below.
```
src/scripts.sh: The demo run scripts of AlignRec.
src/models/alignrec.py: The main model code of AlignRec.
src/configs/model/ALIGNREC.yaml: The config file of AlignRec.
```

To run our code, please run
```commandline
cd src
bash scripts.sh
```

## New Multimodal Features
In our paper, we produce multimodal features which are expected to be better than the commonly used features in Amazon dataset mentioned in MMRec.
Our new multimodal features can be found in ```/data/beit3_128token_add_title_brand_to_text/``` folder.
They represent the aligned multimodal features which fuse image and text information. 
Please note that we still use ```image_feat.npy``` to name them in order to keep consistent with the original multimodal features.


**Please consider to cite our paper if this code helps you, thanks:**
```
@inproceedings{liu2024alignrec,
author = {Yifan Liu, Kangning Zhang, Xiangyuan Ren, Yanhua Huang, Jiarui Jin, Yingjie Qin, Ruilong Su, Ruiwen Xu, Yong Yu, and Weinan Zhang},
title = {AlignRec: Aligning and Training in Multimodal Recommendations},
booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24)},
year = {2024}
}

