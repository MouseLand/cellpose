import torch
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = True
from torch import nn 
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, backbone="vit_l", ps=8, bsize = 256):
        super(Transformer, self).__init__()
        checkpoint = None

        print("checkpoint: ", checkpoint)
        self.encoder = sam_model_registry[backbone](checkpoint).image_encoder
        w = self.encoder.patch_embed.proj.weight.detach()
        nchan = w.shape[0]
        self.ps = ps
        self.encoder.patch_embed.proj = nn.Conv2d(3, nchan, stride=ps, kernel_size=ps)
        self.encoder.patch_embed.proj.weight.data = w[:,:,::16//ps,::16//ps]
        ds = (1024 // 16) // (bsize // ps)
        self.encoder.pos_embed = nn.Parameter(self.encoder.pos_embed[:,::ds,::ds], requires_grad=True)

        self.nout = 3
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)
        
        for blk in self.encoder.blocks:
            blk.window_size = 0
        
    def forward(self, x):        
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        
        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
        
        return x1, torch.randn((x.shape[0], 256), device=x.device)
    
    def load(self, PATH, device, multigpu = True, strict = False):        
        if multigpu:
            state_dict = torch.load(PATH, map_location = device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict = strict)
        else:
            self.load_state_dict(torch.load(PATH), strict = strict)

