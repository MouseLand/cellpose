"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn 
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, backbone="vit_l", ps=8, nout=3, bsize=256, rdrop=0.4,
                  checkpoint=None):
        super(Transformer, self).__init__()

        # instantiate the vit model, default to not loading SAM
        # checkpoint = sam_vit_l_0b3195.pth is standard pretrained SAM
        self.encoder = sam_model_registry[backbone](checkpoint).image_encoder
        w = self.encoder.patch_embed.proj.weight.detach()
        nchan = w.shape[0]
        
        # change token size to ps x ps
        self.ps = ps
        self.encoder.patch_embed.proj = nn.Conv2d(3, nchan, stride=ps, kernel_size=ps)
        self.encoder.patch_embed.proj.weight.data = w[:,:,::16//ps,::16//ps]
        
        # adjust position embeddings for new bsize and new token size
        ds = (1024 // 16) // (bsize // ps)
        self.encoder.pos_embed = nn.Parameter(self.encoder.pos_embed[:,::ds,::ds], requires_grad=True)

        # readout weights for nout output channels
        # if nout is changed, weights will not load correctly from pretrained Cellpose-SAM
        self.nout = nout
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)

        # W2 reshapes token space to pixel space, not trainable
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)
        
        # fraction of layers to drop at random during training
        self.rdrop = rdrop

        # average diameter of ROIs from training images from fine-tuning 
        self.diam_labels = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        # average diameter of ROIs during main training
        self.diam_mean = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        
        # set attention to global in every layer
        for blk in self.encoder.blocks:
            blk.window_size = 0

    def forward(self, x):      
        # same progression as SAM until readout
        x = self.encoder.patch_embed(x)
        
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        
        if self.training and self.rdrop > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (torch.rand((len(x), nlay), device=x.device) < 
                     torch.linspace(0, self.rdrop, nlay, device=x.device)).float()
            for i, blk in enumerate(self.encoder.blocks):            
                mask = rdrop[:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1-mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        # readout is changed here
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
        
        # maintain the second output of feature size 256 for backwards compatibility
           
        return x1, torch.randn((x.shape[0], 256), device=x.device)
    
    def load_model(self, PATH, device, strict = False):        
        state_dict = torch.load(PATH, map_location = device, weights_only=True)
        keys = [k for k in state_dict.keys()]
        if keys[0][:7] == "module.":
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict = strict)
        else:
            self.load_state_dict(state_dict, strict = strict)

    
    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)



class CPnetBioImageIO(Transformer):
    """
    A subclass of the CP-SAM model compatible with the BioImage.IO Spec.

    This subclass addresses the limitation of CPnet's incompatibility with the BioImage.IO Spec,
    allowing the CPnet model to use the weights uploaded to the BioImage.IO Model Zoo.
    """

    def forward(self, x):
        """
        Perform a forward pass of the CPnet model and return unpacked tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        output_tensor, style_tensor, downsampled_tensors = super().forward(x)
        return output_tensor, style_tensor, *downsampled_tensors
    

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(self.nout)
            state_dict = torch.load(filename, map_location=torch.device("cpu"), 
                                    weights_only=True)

        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the model.

        This method overrides the default `load_state_dict` to handle Cellpose's custom
        loading mechanism and ensures compatibility with BioImage.IO Core.

        Args:
            state_dict (Mapping[str, Any]): A state dictionary to load into the model
        """
        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            super().load_state_dict(
                {name: param for name, param in state_dict.items()},
                strict=False)


    

