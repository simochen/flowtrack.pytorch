# flowtrack.pytorch
Pytorch implementation of [FlowTrack](https://arxiv.org/pdf/1804.06208.pdf).

**Simple Baselines for Human Pose Estimation and Tracking** (https://arxiv.org/pdf/1804.06208.pdf)



### TO DO:

- [x] Human detection
- [x] Single person pose estimation
- [x] Optical flow estimation
- [x] Box propagation
- [ ] Pose tracking



### Requirements

```
pytorch >= 0.4.0
torchvision
pycocotools
tensorboardX
```




### Installation

```shell
cd lib
./make.sh
```

Disable cudnn for batch_norm:

```
# PYTORCH=/path/to/pytorch
# for pytorch v0.4.0
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
# for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

### Training

#### Pose Estimation

Download [data folder](https://drive.google.com/drive/folders/1RqsY3ONtYlxQfLg2acJm36qTtqvHDCik?usp=sharing) as `$ROOT/data`. 

```shell
python ./tools/pose/main.py
```

The official code is released on [Microsoft/human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch).

### 

### Demo

#### Pose Estimation

`#TODO`

#### Detection

Download pretrained [detection model](https://drive.google.com/file/d/18PKsPqSBx7C940zz95siAKiI_6aSE5hO/view?usp=sharing) into `models/detection/`. Refer to [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) for more information.

```shell
python ./tools/detection/demo.py
```

#### 

#### Optical Flow Estimation

Download pretrained [flownet](https://drive.google.com/file/d/17d0x6q3FZZCfHMz7vND8E78WIZreqito/view?usp=sharing) into `models/flownet/`. Refer to [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) for more information.

```shell
python ./tools/flownet/demo.py --model </path/to/model>
```



### Update

**2018.12.05:**

-  Add Pose Estimation Models
  - Deconv DenseNet
  - Stacked Hourglass Network
  - FPN