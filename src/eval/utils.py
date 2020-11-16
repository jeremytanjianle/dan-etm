import torch 
import numpy as np

def tensor_2_idx(tensor):
    """
    Converts torch tensors into their respective idx
    """
    tensor = tensor.cpu().detach().numpy() if type(tensor)==torch.Tensor else tensor
    idxes = np.arange(tensor.shape[-1]).reshape(-1,1)

    list_of_idxes = []

    while sum(tensor)>0:
        signed = np.sign(tensor)
        idxes_to_include = (signed * idxes.T)[(signed * idxes.T)!=0].tolist()
        list_of_idxes.extend(idxes_to_include)

        tensor -= signed
        tensor[tensor<0]=0

    return list_of_idxes