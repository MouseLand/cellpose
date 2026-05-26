import time, os
import numpy as np
from cellpose import io, models, train
from pathlib import Path
import torch
import time
import argparse
import time

class TorchTracemalloc():

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = self.end-self.begin
        self.peaked = self.peak-self.begin
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")



if __name__ == "__main__":
    # arg parsing
    io.logger_setup()
    parser = argparse.ArgumentParser(description="Cellpose Command Line Parameters")

    # misc settings
    parser.add_argument("--use_3D", action="store_true")
    parser.add_argument("--ntiles", default=1, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--alg", type=str, default="cpsam", 
                        choices=["cpsam", "cpdino", "cpdino-vitb", "cyto3"])
    args = parser.parse_args()

    alg = args.alg
    ntiles = args.ntiles
    if alg == "cyto3":
        model = models.CellposeModel(device=torch.device("cuda:0"), model_type="cyto3") 
    else:
        if alg == "cpsam":
            alg = "cpsam_v2"
        model = models.CellposeModel(device=torch.device("cuda:0"), pretrained_model=alg)
    if not args.train:
        if args.use_3D:
            if Path("/media/carsen/disk1/datasets_cellpose/cells_3D/tim_nov4_crop2.tif").exists():
                img = io.imread("/media/carsen/disk1/datasets_cellpose/cells_3D/tim_nov4_crop2.tif")
            else:
                img = io.imread("/groups/stringer/stringerlab/datasets_cellpose/cells_3D/tim_nov4_crop2.tif")
            if ntiles==1:
                img = img[50:200, :, 100:250, 100:250].transpose(0,2,3,1).copy()
            else:
                img = img[50:350, :, 50:350, 50:350].transpose(0,2,3,1).copy()
                if ntiles > 2:
                    img = np.tile(img, (ntiles//2, ntiles//2, ntiles//2, 1))
            print("IMG SHAPE: ", img.shape)
            masks = model.eval(img[100,:150,:150], batch_size=32, diameter=30.)[0]
            tic = time.time()
            masks = model.eval(img, do_3D=True, batch_size=32, z_axis=0, 
                               channel_axis=-1, diameter=30.)[0]
            print(f"{time.time() - tic : .4f}")
        else:
            img = io.imread(Path.home() / ".cellpose/data/2D" / "rgb_2D.png")
            if ntiles==1:
                img = img[100:250, 100:250]
            else:
                img = img[50:350, 50:350]
                if ntiles > 2:
                    img = np.tile(img, (ntiles//2, ntiles//2, 1))
            print("IMG SHAPE: ", img.shape)
            masks = model.eval(img[:150, :150], batch_size=32, diameter=30.)[0]
            tic = time.time()
            masks = model.eval(img, batch_size=32, diameter=30.)[0]
            print(f"{time.time() - tic : .4f}")
    else:
        imgs = [io.imread(Path.home() / ".cellpose/data/2D" / "rgb_2D.png"), 
                io.imread(Path.home() / ".cellpose/data/2D" / "gray_2D.png")]
        masks = [io.imread(Path.home() / ".cellpose/data/2D" / "rgb_2D_cp4_gt_masks.png"), 
                 io.imread(Path.home() / ".cellpose/data/2D" / "gray_2D_cp4_gt_masks.png")]
        net = model.net 
        tic = time.time()    
        train.train_seg(net, train_data=imgs, train_labels=masks, learning_rate=1e-5,
                        n_epochs=100, batch_size=1)
        print(f"{time.time() - tic : .4f}")

    #print(torch.cuda.max_memory_allocated(1) / 1e9)
    print(f"{torch.cuda.max_memory_allocated(0) / 1e9 : .4f}")

    