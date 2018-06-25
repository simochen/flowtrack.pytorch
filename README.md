# flowtrack.pytorch
Pytorch implementation of [FlowTrack](https://arxiv.org/pdf/1804.06208.pdf).

**Simple Baselines for Human Pose Estimation and Tracking** (https://arxiv.org/pdf/1804.06208.pdf)



### TO DO:

- [x] Human detection
- [ ] Single person pose estimation
	- [x] Building
	- [ ] Training
- [x] Optical flow estimation
- [x] Box propagation
- [ ] Pose tracking




### Installation

```shell
cd lib
./make.sh
```



### Demo

#### Detection

Download pretrained [detection model](https://drive.google.com/file/d/18PKsPqSBx7C940zz95siAKiI_6aSE5hO/view?usp=sharing) into `models/detection/`. Refer to [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) for more information.

```shell
python ./tools/detection/demo.py
```

#### Optical Flow Estimation

Download pretrained [flownet](https://drive.google.com/file/d/17d0x6q3FZZCfHMz7vND8E78WIZreqito/view?usp=sharing) into `models/flownet/`. Refer to [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) for more information.

```shell
python ./tools/flownet/demo.py --model </path/to/model>
```

