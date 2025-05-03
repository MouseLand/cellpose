from segment_anything import sam_model_registry
from cellpose import io, metrics, models, utils, train, transforms, resnet_torch, dynamics
import time
from tqdm import trange
from torch import nn 
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import os

torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda")

class TransformerMP(nn.Module):
    def __init__(self, backbone="vit_l", ps=8, bsize = 256, rdrop=0.4):
        super(TransformerMP, self).__init__()
        #bsize = 256
        checkpoint = None

        print("checkpoint: ", checkpoint)
        self.encoder = sam_model_registry[backbone](checkpoint).image_encoder
        w = self.encoder.patch_embed.proj.weight.detach()
        nchan = w.shape[0]
        self.ps = ps
        self.encoder.patch_embed.proj = nn.Conv2d(3, nchan, stride=ps, kernel_size=ps).to(device)
        self.encoder.patch_embed.proj.weight.data = w[:,:,::16//ps,::16//ps]
        ds = (1024 // 16) // (bsize // ps)
        self.encoder.pos_embed = nn.Parameter(self.encoder.pos_embed[:,::ds,::ds], requires_grad=True)

        self.nout = 3
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)#.to(device)
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)
        
        self.encoder.size_embed = nn.Parameter(0/32**.5 * torch.randn(32,1024), requires_grad=True)
        self.diam_labels =  nn.Parameter(30.*torch.ones(1), requires_grad=False)
        self.diam_mean = nn.Parameter(30.*torch.ones(1), requires_grad=False)
        self.mkldnn = False 
        self.rdrop = rdrop 

        for blk in self.encoder.blocks:
            blk.window_size = 0

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device
        
    def forward(self, x):
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

        x = self.out(x)
        x = F.conv_transpose2d(x, self.W2, stride = self.ps, padding = 0)
        return x, torch.randn((x.shape[0], 256), device=x.device)
    
    def load_model(self, fname, multigpu=True, strict = True):
        if multigpu:
            state_dict = torch.load(fname, map_location = device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict = strict)
        else:
            self.load_state_dict(torch.load(fname), strict = strict)

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

def train_subsets(root, seed=0, ntrain=1,
                  transformer = False, save_results = True, 
                  save_test_masks = False, keep_model=False,
                  device=torch.device("cuda")):
    io.logger_setup()
    print(device)
    print(f"seed = {seed}, ntrain = {ntrain}")
    mtype = "sam" if transformer else "cyto3"
    netstr = f"{mtype}_seed_{seed}_ntrain_{ntrain}"
    
    train_files = (root / "train").glob("*.tif")
    train_files = natsorted([tf for tf in train_files if "_masks" not in str(tf)])
    ntrain = len(train_files) if ntrain == 0 else ntrain

    np.random.seed(seed + 5)
    itrain = np.random.permutation(len(train_files))[:ntrain]
    train_files = [train_files[i] for i in itrain]

    test_files = (root / "test").glob("*.tif")  
    test_files = natsorted([tf for tf in test_files if "_masks" not in str(tf)])
    print(f"nimg_train = {len(train_files)}, nimg_test = {len(test_files)}")

    train_data, test_data = [], []
    k = 0 
    print("loading images")
    for k in range(2 if len(test_files) > 0 else 1):
        tf = train_files if k==0 else test_files
        for i in trange(len(tf)):
            img = io.imread(tf[i])
            img = np.tile(img[np.newaxis,:,:], (3,1,1))
            img[1:] = 0
            if k==0:
                train_data.append(img)
            else:
                test_data.append(img)
    print("loading labels and computing flows")
    train_masks = [io.imread(str(train_files[i])[:-4] + f'_masks.tif') for i in trange(len(train_files))]
    train_labels = dynamics.labels_to_flows(train_masks, device=torch.device("cuda"))
    if len(test_files) > 0:
        test_masks = [io.imread(str(test_files[i])[:-4] + f'_masks.tif') for i in trange(len(test_files))]
        test_labels = dynamics.labels_to_flows(test_masks, device=torch.device("cuda"))
    else:
        test_data = None
        test_labels = None
        
    if transformer:
        ps = 8
        backbone = "vit_l"
        rdrop = 0.4
        net = TransformerMP(ps=ps, backbone=backbone, rdrop=rdrop).to(device)
        net.load_model("models/cpsam8_0_2100_8_402175188", strict=False, multigpu=False)

    else:
        net = resnet_torch.CPnet(nbase=[2, 32, 64, 128, 256], nout=3, sz=3).to(device)
        net.load_model("models/cyto3", device=device)

    bsize = 256 if transformer else 224
    channels = None if transformer else [1,0]
    
    if ntrain >= 0:
        learning_rate = 1e-5 if transformer else 5e-3
        weight_decay = 0.1 if transformer else 1e-4
        SGD = False 
        soft_start = not transformer
        batch_size = 1 if transformer else 8
        n_epochs = 100 if transformer else 300s

        rescale = not transformer
        scale_range = 0.5 
        max_nimg = 800 
        nimg_per_epoch_min = 8

        # train
        tic = time.time()
        out = train.train_seg(net, train_data=train_data, train_labels=train_labels, 
                              test_data=test_data, test_labels=test_labels,
                            learning_rate=learning_rate, weight_decay=weight_decay,
                            SGD=SGD, batch_size=batch_size, n_epochs=n_epochs,
                            soft_start=soft_start, channels=channels, bsize=bsize,
                            nimg_per_epoch=min(max_nimg, max(nimg_per_epoch_min, len(train_data))), 
                            rescale=rescale, 
                            scale_range=scale_range, save_path=root,
                            nimg_test_per_epoch=len(test_data), model_name=netstr,
                              min_train_masks=0)
    
        toc = time.time() - tic
        print(f"training time: {toc:.2f}s")
        filename, train_losses, test_losses = out
        filename = str(filename)
    else:
        toc = 0
        filename, train_losses, test_losses = None, None, None
    # evaluate
    if ntrain >= 0 or transformer:
        model = models.CellposeModel(gpu=True, nchan=3 if transformer else 2, pretrained_model=None)
        net.eval() 
        model.net = net
        diameter = net.diam_labels.item() if not transformer else 30.
    else:
        model = models.Cellpose(gpu=True, model_type="cyto3")
        diameter = 0

    masks_pred = model.eval(test_data, diameter=diameter, channels=channels, 
                            batch_size=64, bsize=bsize)[0]
    masks_gt = [tl.astype("uint16") for tl in test_masks]
    threshold = np.arange(0.5, 1.0, 0.05)
    ap, tp, fp, fn = metrics.average_precision(masks_gt, masks_pred, threshold=threshold)
    print(ap[:, [0, 5, 8]].mean(axis=0))
    if save_results:
        np.save(root / f"models/{netstr}_AP_TP_FP_FN.npy", {
                "threshold": threshold, "ap": ap, "tp": tp, "fp": fp, "fn": fn,
                "ntrain_masks": len(train_data), "test_files": test_files,
                "diam_labels": diameter, "model_path": filename,
                "train_losses": train_losses, "test_losses": test_losses,
                "test_masks_pred": masks_pred if save_test_masks else None,
                "train_time": toc,
            "nrois": np.array([td.max() for td in train_masks]).sum()})
        
    if ntrain>=0 and not keep_model:
        os.remove(filename)
    #if save_test_masks:
    #    for i in range(len(test_files)):
    #        fname = os.path.splitext(str(test_files[i]))[0]
    #        io.imsave(fname + "_" + netstr + "_masks.tif", masks_pred[i])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transformer", type=int, default=0)
    
    args = parser.parse_args()
    ntrain = args.ntrain
    seed = args.seed
    transformer = args.transformer
    root = Path(args.root)
    train_subsets(root, ntrain=ntrain, seed=seed, transformer=args.transformer,
                  save_test_masks=True if seed==0 else False,
                  keep_model=True)

