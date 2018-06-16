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

Download pretrained [detection model]() into `models/detection/`.

```shell
python ./tools/detection/demo.py
```

#### Optical Flow Estimation

Download pretrained [flownet]() into `models/flownet/`.

```shell
python ./tools/flownet/demo.py --model </path/to/model>
```

