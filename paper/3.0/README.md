### cellpose 3 paper

`analysis.py` provides the analysis functions for the paper, and `figures.py` contains the code to create the figures.

To train the super-generalist cyto3 segmentation model, we did not perform any image processing on the external datasets other than normalizing the image per channel. 
The livecell masks contain overlaps, so we removed overlaps with this [function](https://github.com/MouseLand/cellpose/blob/ae795e0f95cb2ecfc3e5b24185a2370bd2cd2225/paper/2.0/datasets.py#L271). 
The training command for the model is as follows (assuming the tiffs are normalized per channel already):

python -m cellpose --file_list probs_generalist.npy --dir /media/carsen/ssd4/datasets_cellpose/ --verbose --train --min_train_masks 0 --chan 1 --chan2 2 --pretrained_model None --no_norm --nimg_per_epoch 800 --nimg_test_per_epoch 100 --SGD 0 --learning_rate 0.005 --n_epochs 5000 --use_gpu --train_size

See the example probs_generalist.npy file [here](https://www.cellpose.org/static/data/probs_generalist.npy) for how to structure cellpose to use filelist as an input for training. If you're on windows, you may need to set the pathlib path before loading the npy:
```
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np

dat = np.load("probs_generalist.npy", allow_pickle=True).item()
print(dat.keys())
```

