

# HOTR: End-to-End Human-Object Interaction Detection with Transformers

This paper reproduces a paper from CVPR2021. The author has provided the source code, and I have completed two tasks based on the author's code.

- Use the newly designed POS-Decoder to replace the author's original Decoder to improve the stability of Hungarian matching
- Added image visualization code to facilitate viewing the prediction results of the model

We can see the visualization results in the [visual] file.
The detail of using the code is similar as author. The following is the part of author's original README.md. The Result is the result of my experiment.
## 1. Environmental Setup
```bash
$ conda create -n kakaobrain python=3.7
$ conda install -c pytorch pytorch torchvision # PyTorch 1.7.1, torchvision 0.8.2, CUDA=11.0
$ conda install cython scipy
$ pip install pycocotools
$ pip install opencv-python
$ pip install wandb
```
## 2. HOI dataset setup
Our current version of HOTR supports the experiments for both [V-COCO](https://github.com/s-gupta/v-coco) and [HICO-DET](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) dataset.
Download the dataset under the pulled directory.
For HICO-DET, we use the [annotation files](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) provided by the PPDM authors.
Download the [list of actions](https://drive.google.com/open?id=1EeHNHuYyJI-qqDk_-5nay7Mb07tzZLsl) as `list_action.txt` and place them under the unballed hico-det directory.
Below we present how you should place the files.
```bash
# V-COCO setup
$ git clone https://github.com/s-gupta/v-coco.git
$ cd v-coco
$ ln -s [:COCO_DIR] coco/images # COCO_DIR contains images of train2014 & val2014
$ python script_pick_annotations.py [:COCO_DIR]/annotations

# HICO-DET setup
$ tar -zxvf hico_20160224_det.tar.gz # move the unballed folder under the pulled repository

# dataset setup
HOTR
 │─ v-coco
 │   │─ data
 │   │   │─ instances_vcoco_all_2014.json
 │   │   :
 │   └─ coco
 │       │─ images
 │       │   │─ train2014
 │       │   │   │─ COCO_train2014_000000000009.jpg
 │       │   │   :
 │       │   └─ val2014
 │       │       │─ COCO_val2014_000000000042.jpg
 :       :       :
 │─ hico_20160224_det
 │       │─ list_action.txt
 │       │─ annotations
 │       │   │─ trainval_hico.json
 │       │   │─ test_hico.json
 │       │   └─ corre_hico.npy
 :       :
```

If you wish to download the datasets on our own directory, simply change the 'data_path' argument to the directory you have downloaded the datasets.
```bash
--data_path [:your_own_directory]/[v-coco/hico_20160224_det]
```


## 3. How to Train/Test HOTR
For both training and testing, you can either run on a single GPU or multiple GPUs.
```bash
# single-gpu training / testing
$ make [vcoco/hico]_single_[train/test]

# multi-gpu training / testing (4 GPUs)
$ make [vcoco/hico]_multi_[train/test]
```
For testing, you can either use your own trained weights and pass the group name and run name to the 'resume' argument, or use our provided weights.
Below is the example of how you should edit the Makefile.
```bash
# [Makefile]
# Testing your own trained weights
[vcoco/hico]_multi_test:
  python -m torch.distributed.launch \
		--nproc_per_node=8 \
    ...
    --resume checkpoints/[vcoco/hico_det]/[:group_name]/[:run_name]/best.pth # the best performing checkpoint is saved in this format

# Testing our provided trained weights
[vcoco/hico]_multi_test:
  python -m torch.distributed.launch \
		--nproc_per_node=8 \
    ...
    --resume checkpoints/[vcoco/hico_det]/[vcoco/hico]_q16.pth # download the q16.pth as described below.
```
In order to use our provided weights, you can download the weights provided below.
Then, pass the directory of the downloaded file (for example, to test our pre-trained weights on the vcoco dataset, we put the downloaded weights under the directory checkpoints/vcoco/vcoco_q16.pth) to the 'resume' argument.

## 4. Result
By adding the component of POS-Decoder, the model can achieve better results on the vcoco dataset. The following table shows the results of the model on the vcoco dataset.
### VCOCO-dataset
Limited by the number of GPUs, I only trained the model in the vcoco dataset. The following table shows the results of the model on the vcoco dataset.

| Epoch | Model    | Queries | Interaction Decoder | Scenario 1 | Scenario 2 |
|-------|----------|---------|---------------------|------------|------------|
| 100   | HOTR     | 16      | Normal Decoder      | 55.2       | 64.4       |
| 100   | POS-HOTR | 16      | Pos Decoder         | 60.5       | 65.6       |

