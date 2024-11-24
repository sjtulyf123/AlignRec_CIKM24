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
@inproceedings{10.1145/3627673.3679626,
author = {Liu, Yifan and Zhang, Kangning and Ren, Xiangyuan and Huang, Yanhua and Jin, Jiarui and Qin, Yingjie and Su, Ruilong and Xu, Ruiwen and Yu, Yong and Zhang, Weinan},
title = {AlignRec: Aligning and Training in Multimodal Recommendations},
year = {2024},
isbn = {9798400704369},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627673.3679626},
doi = {10.1145/3627673.3679626},
pages = {1503–1512},
numpages = {10},
location = {Boise, ID, USA},
series = {CIKM '24}
}

