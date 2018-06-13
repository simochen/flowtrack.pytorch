## Optical Flow Estimation

This part of code is mainly based on [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch).

### Installation

```shell
# install custom layers
cd flowtrack.pytorch/optical_flow/
bash install.sh
```

### Inference

```
python demo.py --model FlowNet2S --resume </path/to/checkpoint>
```

