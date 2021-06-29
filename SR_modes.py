import torch
import numpy as np
from lib.utils import load_img, imresize

def run_SR(SR_model, images, sparse=1, factor=2):
    ''' inputs 
            SR_model - Spatial SR model
            images - target images
            sparse - sparse = 1, defualt for spatial SR
                     sparse = 2, only use for Joint SAR
            factor - scale factor
    '''
    index = np.reshape(range(81), [9, 9])
    input_index = index[::sparse, ::sparse].flatten()

    h, w = images[0].shape

    h -= h % factor
    w -= w % factor

    images = [images[id][:h, :w] for id in input_index]

    if sparse == 1:
        le = 9
    elif sparse == 2:
        le = 5
    else:
        assert("error")
    outputs = []
    for i in range(le):
        for j in range(le):
            central_id = 9*sparse*i+sparse*j
            inputs, is_coner = load_img(images, central_id, sparse, factor)
            output = SR_model(inputs, [[0, is_coner]])
            outputs.append(output[0, 0].detach().cpu().numpy())
    return outputs, images
