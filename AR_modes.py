import torch
import torch.nn as nn
import numpy as np
from lib.utils import load_img, imresize, toTensor
from skimage.measure import compare_ssim, compare_psnr

MSE = nn.MSELoss()

def run_AR(AR_model, SR_model, images, device):
    ''' inputs 
            AR_model - Angular SR model
            SR_model - Spatial SR model where only flownet is used
            images - target images
            device - device to run
    '''

    index = np.reshape(range(81), [9, 9])
    input_index = index[::2, ::2].flatten()
    target_index_h = index[::2, 1::2].flatten() #index
    target_index_v = index[1::2, ::2].flatten() #index3
    target_index_c = index[1::2, 1::2].flatten() #index2
    

    inputs = np.reshape(toTensor(images, input_index, device), [5, 5])
    targets_h = np.reshape(toTensor(images, target_index_h, device), [5, 4])
    targets_v = np.reshape(toTensor(images, target_index_v, device), [4, 5])
    targets_c = np.reshape(toTensor(images, target_index_c, device), [4, 4])

    psnr = []
    ssim = []
    with torch.no_grad():
        for i in range(5):
            for j in range(4):
                # Cross
                if i < 4:
                    output = Cross(AR_model, SR_model,
                        inputs[i, j], inputs[i+1, j+1],
                        inputs[i, j+1], inputs[i+1, j])
                    output = np.squeeze(output.detach().cpu().numpy())
                    target = np.squeeze(targets_c[i, j].detach().cpu().numpy())
                    psnr.append(compare_psnr(output, target))
                    ssim.append(compare_ssim(output, target))
                # Horizontal
                output = Horizontal(AR_model, SR_model, inputs[i, j], inputs[i, j+1])
                output = np.squeeze(output.detach().cpu().numpy())
                target = np.squeeze(targets_h[i, j].detach().cpu().numpy())
                psnr.append(compare_psnr(output, target))
                ssim.append(compare_ssim(output, target))
                # Vertical
                output = Vertical(AR_model, SR_model, inputs[j, i], inputs[j+1, i])
                output = np.squeeze(output.detach().cpu().numpy())
                target = np.squeeze(targets_v[j, i].detach().cpu().numpy())
                psnr.append(compare_psnr(output, target))
                ssim.append(compare_ssim(output, target))
    
    return psnr, ssim


def Cross(model, flow, inputs0, inputs1, inputs2, inputs3):
    F_0_1 = flow.Flow_forward(torch.cat((inputs0, inputs1), dim=1))
    F_1_0 = flow.Flow_forward(torch.cat((inputs1, inputs0), dim=1))
    F_2_3 = flow.Flow_forward(torch.cat((inputs2, inputs3), dim=1))
    F_3_2 = flow.Flow_forward(torch.cat((inputs3, inputs2), dim=1))
    
    # AR
    F_t_0 = 0.5 * F_1_0
    F_t_1 = 0.5 * F_0_1
    F_t_3 = 0.5 * F_2_3
    F_t_2 = 0.5 * F_3_2

    flows_ = [F_t_0, F_t_1, F_t_2, F_t_3]
    inputs_ = [inputs0, inputs1, inputs2, inputs3]
    return model(inputs_, 0, flows_)

def Horizontal(model, flow, inputs0, inputs1):
    F_0_1 = flow.Flow_forward(torch.cat((inputs0, inputs1), dim=1))
    F_1_0 = flow.Flow_forward(torch.cat((inputs1, inputs0), dim=1))
    
    # AR
    F_t_0 = 0.5 * F_1_0
    F_t_1 = 0.5 * F_0_1
    
    flows_ = [F_t_0, F_t_1, 
              torch.zeros_like(F_t_1), torch.zeros_like(F_t_1)]
    inputs_ = [inputs0, inputs1,
               torch.zeros_like(inputs0),
               torch.zeros_like(inputs0)]
    return model(inputs_, 1, flows_)


def Vertical(model, flow, inputs0, inputs1):
    F_0_1 = flow.Flow_forward(torch.cat((inputs0, inputs1), dim=1))
    F_1_0 = flow.Flow_forward(torch.cat((inputs1, inputs0), dim=1))
    
    # AR
    F_t_0 = 0.5 * F_1_0
    F_t_1 = 0.5 * F_0_1
    
    flows_ = [F_t_0, torch.zeros_like(F_t_1), 
              F_t_1, torch.zeros_like(F_t_1)]
    inputs_ = [inputs0, torch.zeros_like(inputs0),
               inputs1, torch.zeros_like(inputs0)]
    return model(inputs_, 2, flows_)
