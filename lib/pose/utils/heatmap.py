from __future__ import absolute_import

import os
import math
import numpy as np
import torch

def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return torch.from_numpy(h)

def draw_gaussian(img, pt, sigma):
    # Draw a 2D gaussian
    threshold = math.ceil(3 * sigma)   # sigma * sqrt(2*ln(100))
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - threshold), int(pt[1] - threshold)]
    br = [int(math.ceil(pt[0] + threshold)), int(math.ceil(pt[1] + threshold))]
    if (ul[0] >= img.size(1) or ul[1] >= img.size(0) or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img
    # # Generate gaussian
    # size = 2 * threshold + 1
    # x = np.arange(0, size, 1, float)
    # y = x[:, np.newaxis]
    # x0 = y0 = threshold
    # # The gaussian is not normalized, we want the center value to equal 1
    # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # g = torch.from_numpy(g)
    #
    # # Usable gaussian range
    # g_x = [max(0, -ul[0]), min(br[0]+1, img.size(1)) - ul[0]]
    # g_y = [max(0, -ul[1]), min(br[1]+1, img.size(0)) - ul[1]]
    # # Image range
    # img_x = [max(0, ul[0]), min(br[0]+1, img.size(1))]
    # img_y = [max(0, ul[1]), min(br[1]+1, img.size(0))]
    #
    # torch.max(img[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
    #         out = img[img_y[0]:img_y[1], img_x[0]:img_x[1]])

    # Image range
    img_x = [max(0, ul[0]), min(br[0]+1, img.size(1))]
    img_y = [max(0, ul[1]), min(br[1]+1, img.size(0))]
    # Generate gaussian
    x = np.arange(img_x[0], img_x[1])
    y = np.arange(img_y[0], img_y[1])[:,np.newaxis]
    g = np.exp(- ((x - pt[0]) ** 2 + (y - pt[1]) ** 2) / (2.0 * sigma ** 2))
    g = g.float()

    torch.max(img[img_y[0]:img_y[1], img_x[0]:img_x[1]], g,
           out = img[img_y[0]:img_y[1], img_x[0]:img_x[1]])

    return img

def point_to_seg(pt, seg):
    cross = (pt[0]-seg[0][0])*(seg[1][0]-seg[0][0])+(pt[1]-seg[0][1])*(seg[1][1]-seg[0][1])
    if cross <= 0:
        return (pt[0]-seg[0][0])**2 + (pt[1]-seg[0][1])**2
    seg_d2 = (seg[1][0]-seg[0][0])**2 + (seg[1][1]-seg[0][1])**2
    if cross >= seg_d2:
        return (pt[0]-seg[1][0])**2 + (pt[1]-seg[1][1])**2
    r = cross / seg_d2
    q = [seg[0][0] + r*(seg[1][0]-seg[0][0]), seg[0][1] + r*(seg[1][1]-seg[0][1])]   # seg[1] + r * (seg[2]-seg[1])
    return (pt[0]-q[0])**2 + (pt[1]-q[1])**2

# def point_map_to_seg(pt_map, seg):
#     # pt_map: h x w x 2
#     # seg: 2 x 2  => list, tuple, ...
#     h, w, _ = pt_map.size()
#     dist_map = torch.zeros(h, w)
#     p1_map = torch.Tensor(seg[0]).expand(h, w, 2)
#     p2_map = torch.Tensor(seg[1]).expand(h, w, 2)
#     sub_p1_map = pt_map.sub(p1_map)
#     # cross: (p-p1).dot(p2-p1)
#     cross = sub_p1_map[:,:,0].mul(seg[1][0]-seg[0][0]).add(sub_p1_map[:,:,1].mul(seg[1][1]-seg[0][1]))
#     mask_1 = cross.le(0)    # p1 side
#     dist_map.masked_scatter_(mask_1, sub_p1_map.pow(2).sum(2))
#     # seg_d2: ||p2-p1||^2
#     seg_d2 = (seg[1][0]-seg[0][0])**2 + (seg[1][1]-seg[0][1])**2
#     mask_2 = cross.ge(seg_d2)   # p2 side
#     dist_map.masked_scatter_(mask_2, pt_map.sub(p2_map).pow(2).sum(2))
#     # between p1 and p2
#     mask_3 = mask_1.max(mask_2).eq(0)
#     cross.div_(seg_d2)
#     q_map = torch.addcmul(p1_map, cross.view(h,w,1).expand(h,w,2), p2_map.sub(p1_map))
#     dist_map.masked_scatter_(mask_3, pt_map.sub(q_map).pow(2).sum(2))
#     return dist_map

def point_map_to_seg(pt_map, seg):
    # pt_map: h x w x 2
    # seg: 2 x 2  => list, tuple, ...
    h, w, _ = pt_map.size()
    p1_map = torch.Tensor(seg[0]).expand(h, w, -1)
    p2_map = torch.Tensor(seg[1]).expand(h, w, -1)
    sub_p1_map = pt_map.sub(p1_map)
    # cross: (p-p1).dot(p2-p1)
    cross = sub_p1_map[:,:,0].mul(seg[1][0]-seg[0][0]).add(sub_p1_map[:,:,1].mul(seg[1][1]-seg[0][1]))
    mask_1 = cross.le(0).float()    # p1 side
    dist_map = mask_1.mul(sub_p1_map.pow(2).sum(2))
    # seg_d2: ||p2-p1||^2
    seg_d2 = (seg[1][0]-seg[0][0])**2 + (seg[1][1]-seg[0][1])**2
    mask_2 = cross.ge(seg_d2).float()   # p2 side
    dist_map.add_(mask_2.mul(pt_map.sub(p2_map).pow_(2).sum(2)))
    # between p1 and p2
    mask_3 = mask_1.max(mask_2).eq(0).float()
    seg_d2 = max(seg_d2, 1e-6)
    cross.div_(seg_d2)
    q_map = torch.addcmul(p1_map, cross.view(h,w,1).expand(h,w,2), p2_map.sub(p1_map))
    dist_map.add_(mask_3.mul(pt_map.sub(q_map).pow_(2).sum(2)))
    return dist_map

def draw_line_gaussian(img, seg, sigma):
    # Draw a 2D line gaussian
    threshold = math.ceil(3 * sigma)   # sigma * sqrt(2*ln(100))
    # Check that any part of the gaussian is in-bounds
    ul = [int(min(seg[0][0],seg[1][0]) - threshold), int(min(seg[0][1],seg[1][1]) - threshold)]
    br = [int(math.ceil(max(seg[0][0],seg[1][0]) + threshold)), int(math.ceil(max(seg[0][1],seg[1][1]) + threshold))]
    if (ul[0] >= img.size(1) or ul[1] >= img.size(0) or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img
    # Image range
    img_x = [max(0, ul[0]), min(br[0]+1, img.size(1))]
    img_y = [max(0, ul[1]), min(br[1]+1, img.size(0))]
    # Generate coords map
    x = torch.arange(img_x[0], img_x[1])
    y = torch.arange(img_y[0], img_y[1])
    w = x.size(0)
    h = y.size(0)
    pt_map = torch.stack((x.expand(h,w), y.view(-1,1).expand(h,w)), 2)
    # Compute dist map
    dist_map = point_map_to_seg(pt_map, seg)
    # Generate gaussian map
    gaussian_map = torch.exp(dist_map.div_(2 * sigma ** 2).neg_())

    torch.max(img[img_y[0]:img_y[1], img_x[0]:img_x[1]], gaussian_map,
            out = img[img_y[0]:img_y[1], img_x[0]:img_x[1]])
    return img
