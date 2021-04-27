from scipy.ndimage import convolve
import cv2, torch
import torch.nn as nn
import numpy as np
from math import ceil
from os.path import isfile
from torch.autograd import Variable

list_ = [[0], [range(1, 8)][0], [8], [range(9, 72, 9)][0],
        [range(17, 72, 9)][0], [72], [range(73, 80)][0], [80]]

def BI(inputs, factor=2):
    factor = 1/factor
    return np.float32(imresize(inputs, factor)[None, None, ..., None])

def load_img(images_, central_id=40, sparse = 1, factor=2):
    img_id = central_id - (10*sparse)
    images = []
    for i in range(3):
        for j in range(3):
            try:
                img = images_[9*sparse*i + j*sparse + img_id]
            except:
                img = images_[40]
            images.append(img)
    zeros = np.zeros_like(img)
    is_coner = 0
    coner = []
    if central_id in list_[0]:
        coner = [0, 1, 2, 3, 6]
        is_coner = 1
    elif central_id in list_[1]:
        coner = [0, 1, 2]
        is_coner = 2
    elif central_id in list_[2]:
        coner = [0, 1, 2, 5, 8]
        is_coner = 3
    elif central_id in list_[3]:
        coner = [0, 3, 6]
        is_coner = 4
    elif central_id in list_[4]:
        coner = [2, 5, 8]
        is_coner = 5
    elif central_id in list_[5]:
        coner = [0, 3, 6, 7, 8]
        is_coner = 6
    elif central_id in list_[6]:
        coner = [6, 7, 8]
        is_coner = 7
    elif central_id in list_[7]:
        coner = [2, 5, 6, 7, 8]
        is_coner = 8
    for i in coner:
        images[i] = zeros
    images = [torch.from_numpy(BI(image, factor)).cuda() for image in images]
    images = torch.cat(images, 4)
    return images, is_coner

def rgb2y(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr[..., 0] / 255. #normalize

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, output_shape=None, mode="vec"):
    kernel = cubic
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def init_kernel_SR():
    index = [
        np.concatenate(
            [np.arange(0, 32*9)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(0, 32),
             np.arange(0, 32), np.arange(0, 32), np.arange(32 * 5, 32 * 6),
             np.arange(32 * 7, 32 * 8), np.arange(32 * 7, 32 * 9)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(0, 32),
             np.arange(0, 32), np.arange(32 * 4, 32 * 9)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(0, 32),
             np.arange(0, 32), np.arange(32 * 4, 32 * 5), np.arange(0, 32),
             np.arange(32 * 6, 32 * 8), np.arange(0, 32)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(32 * 2, 32 * 4),
             np.arange(0, 32), np.arange(32 * 5, 32 * 6), np.arange(0, 32),
             np.arange(32 * 7, 32 * 9)]),
        np.concatenate(
            [np.arange(0, 32 * 3), np.arange(0, 32), np.arange(32 * 4, 32 * 5),
             np.arange(0, 32), np.arange(32 * 6, 32 * 8), np.arange(0, 32)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(32 * 2, 32 * 4),
             np.arange(0, 32), np.arange(32 * 5, 32 * 6), np.arange(0, 32),
             np.arange(0, 32), np.arange(0, 32)]),
        np.concatenate(
            [np.arange(0, 32 * 6), np.arange(0, 32),
             np.arange(0, 32), np.arange(0, 32)]),
        np.concatenate(
            [np.arange(0, 32 * 3), np.arange(0, 32),
             np.arange(32*4, 32*5), np.arange(0, 32), np.arange(0, 32),
             np.arange(0, 32), np.arange(0, 32)])]
    pruning = np.zeros([9, 32*9, 32*9])
    for i in range(9):
        pruning[i] = np.eye(32*9)[index[i]]
    return pruning.astype(np.float32)

def init_kernel_AR():
    index = [
        np.concatenate(
            [np.arange(0, 32*4)]),
        np.concatenate(
            [np.arange(0, 32*3), np.arange(0, 32)]),
        np.concatenate(
            [np.arange(0, 32), np.arange(0, 32), np.arange(32, 64), np.arange(32, 64)])]
    pruning = np.zeros([3, 32*4, 32*4])
    for i in range(3):
        pruning[i] = np.eye(32*4)[index[i]]
    return pruning.astype(np.float32)

def mask():
    masks = np.zeros([32*9, 32*9])
    masks[0*32:1*32, :] += np.concatenate([np.ones(32*9)])
    masks[1*32:2*32, :] += np.concatenate([np.ones(32*3), np.zeros(32),
                               np.ones(32), np.zeros(32*4)])
    masks[2*32:3*32, :] += np.concatenate([np.ones(32*6), np.zeros(32*3)])
    masks[3*32:4*32, :] += np.concatenate([np.ones(32), np.zeros(32), np.ones(32*2),
                               np.zeros(32), np.ones(32), np.zeros(32*3)])
    masks[4*32:5*32, :] += np.concatenate([np.ones(32*3), np.zeros(32), np.ones(32),
                               np.zeros(32), np.ones(32*2), np.zeros(32*1)])
    masks[5*32:6*32, :] += np.concatenate([np.ones(32), np.zeros(32), np.ones(32*2),
                               np.zeros(32), np.ones(32), np.zeros(32),
                               np.ones(32*2)])
    masks[6*32:7*32, :] += np.concatenate([np.ones(32), np.zeros(32*3), np.ones(32),
                               np.zeros(32), np.ones(32*2), np.zeros(32)])
    masks[7*32:8*32, :] += np.concatenate([np.ones(32), np.zeros(32*3), np.ones(32*5)])
    masks[8*32:9*32, :] += np.concatenate([np.ones(32), np.zeros(32*4), np.ones(32),
                               np.zeros(32), np.ones(32*2)])
    return masks.astype(np.float32)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border', align_corners=True)

    return output


def conv2(c_in, c2_out, kernel=3, stride=1, padding='same'):
    if padding == 'same':
        pad = (kernel//2, kernel//2)
    else:
        pad = (0, 0)
    return nn.Conv2d(c_in, c2_out, (kernel, kernel),
                     (stride, stride), pad).cuda()
