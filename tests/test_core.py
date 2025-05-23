from cellpose.core import assign_device
import pytest
import torch
from unittest.mock import patch


is_cuda_available = torch.cuda.is_available()

is_directml_available = False
try:
    import torch_directml
    is_directml_available = torch_directml.is_available()
except ImportError:
    pass

is_mps_available = torch.backends.mps.is_available()



@pytest.mark.parametrize(
    "gpu,disable_cuda,disable_mps",
    [
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
    ]
)
def test_assign_device(gpu, disable_cuda, disable_mps):
    if disable_cuda and disable_mps:
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            assigned_device, gpu = assign_device(gpu=gpu)

            if is_directml_available:
                expected_device = torch_directml.device()
            else:
                expected_device = torch.device('cpu')

    elif disable_cuda and not disable_mps:
        with patch('torch.cuda.is_available', return_value=False):
            assigned_device, gpu = assign_device(gpu=gpu)

        if is_mps_available:
            expected_device = torch.device('mps')
        elif is_directml_available:
            expected_device = torch_directml.device()
        else:
            expected_device = torch.device('cpu')


    elif not disable_cuda and disable_mps:
        with patch('torch.backends.mps.is_available', return_value=False):
            assigned_device, gpu = assign_device(gpu=gpu)
        
        if is_cuda_available:
            expected_device = torch.device('cuda')
        elif is_directml_available:
            expected_device = torch_directml.device()
        else:
            expected_device = torch.device('cpu')
    
    elif not disable_cuda and not disable_mps:
        assigned_device, gpu = assign_device(gpu=gpu)
        
        if is_cuda_available:
            expected_device = torch.device('cuda')
        elif is_mps_available:
            expected_device = torch.device('mps')
        elif is_directml_available:
            expected_device = torch_directml.device()
        else:
            expected_device = torch.device('cpu')

    if not gpu:
        expected_device = torch.device('cpu')

    assert assigned_device == expected_device
    