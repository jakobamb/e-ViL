import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter

def _scatter_evil(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary and List as input.
    
    Complete loss of generality -- only use with e-ViL"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    kwargs = tuple(kwargs)

    # check if any inputs are still lists
    for gpu_idx in range(len(target_gpus)):
        b = inputs[gpu_idx][0].shape[0]

        # convert to list
        inputs[gpu_idx] = list(inputs[gpu_idx])

        # for lists, split according to batch size and gpu_idx
        for idx, gpu_input in enumerate(inputs[gpu_idx]):
            if isinstance(gpu_input, list):
                slice_idx = gpu_idx*b
                inputs[gpu_idx][idx] = gpu_input[slice_idx:slice_idx+b]
    inputs = tuple(inputs)

    return inputs, kwargs


class DataParallel_eViL(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel_eViL, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_evil(inputs, kwargs, device_ids, dim=self.dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
