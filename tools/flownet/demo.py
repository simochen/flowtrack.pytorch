from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib
# matplotlib.use('Agg')
import _init_paths
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

import os
import argparse
from model import models
from utils import tools
from utils import flowlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', '-i', default='samples/img0.ppm', type=str, help='first input image')
    parser.add_argument('--input2', '-p', default='samples/img1.ppm', type=str, help='second input image')

    parser.add_argument('--save', '-s', default='results', type=str, help='directory for saving')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--number_gpus', '-ng', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default = 255.)
    
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2S')

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()

        args.model_class = tools.module_to_dict(models)[args.model]
                
    # Dynamically load model with parameters passed in via "--model_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:

        kwargs = tools.kwargs_from_args(args, 'model')
        model = args.model_class(args, **kwargs)

        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_err = checkpoint['best_EPE']
            model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        else:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        # assing to cuda or wrap with dataparallel, model and loss 
        if args.number_gpus > 0:
            block.log('Initializing CUDA')
            model = model.cuda()
            if args.fp16:
                model = model.half()
                param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]

            if args.number_gpus > 1:
                block.log('Parallelizing')
                model = torch.nn.DataParallel(model, device_ids=list(range(args.number_gpus)))

        else:
            block.log('CUDA not being used')

    model.eval()
    # Prepare img pair
    # H x W x 3(RGB)
    im1 = imread(args.input1)
    im2 = imread(args.input2)
    # B x 3(RGB) x 2(pair) x H x W
    ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
    ims = torch.from_numpy(ims)
    ims_v = Variable(ims, volatile=True).cuda()
    # B x 2 x H x W
    pred_flow = model(ims_v).cpu().data
    pred_flow = pred_flow[0].numpy().transpose((1,2,0))	# H x W x 2
    flowlib.write_flow(pred_flow, os.path.join(args.save, 'output.flo'))
    flow_im = flowlib.flow_to_image(pred_flow)

    # Visualization
    # plt.imshow(flow_im)
    # plt.savefig(os.path.join(args.save, 'flow.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(args.save, 'flow.png'))
    plt.imsave(os.path.join(args.save, 'flow.png'), flow_im)
