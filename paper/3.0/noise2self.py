"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import transforms, io, metrics
from cellpose.models import CellposeModel

# uses torch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torch.nn import MSELoss
from torch.optim import Adam

# PATH TO REPO
import sys

sys.path.append("/github/noise2self/")
from mask import Masker
from models.unet import Unet

device = torch.device("cuda")


class Cells(Dataset):

    def __init__(self, data, xy=(128, 128)):
        self.data = data
        self.cell_transforms = v2.Compose([
            v2.RandomRotation(degrees=180),
            v2.RandomResizedCrop(size=xy, scale=(0.75, 1.25), ratio=(1.0, 1.0),
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.cell_transforms(torch.from_numpy(self.data[index]).to(device))
        return img


def train_per_image(img_noisy):

    masker = Masker(width=4, mode='interpolate')
    torch.cuda.manual_seed(0)
    model = Unet().to(device)

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-4)

    cells_train = Cells(np.tile(img_noisy[np.newaxis, ...], (8, 1, 1, 1)),
                        xy=(128, 128))
    data_loader = DataLoader(cells_train, batch_size=8, shuffle=True)

    for ep in range(100):
        for k, batch in enumerate(data_loader):
            noisy_images = batch

            net_input, mask = masker.mask(noisy_images, k)
            net_output = model(net_input)

            loss = loss_function(net_output * mask, noisy_images * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    img = img_noisy.copy()
    img, ysub, xsub = transforms.pad_image_ND(img)
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        simple_output = model(img)
    out = simple_output.squeeze()[ysub[0]:ysub[-1] + 1,
                                  xsub[0]:xsub[-1] + 1].cpu().numpy()

    return out


def train_per_image_synthetic(root, ctype="cyto2", plot=False, save=True):
    noise_type = "poisson"

    dat = np.load(root / "noisy_test" / f"test_{noise_type}.npy",
                  allow_pickle=True).item()
    test_noisy = dat["test_noisy"]
    test_labels = dat["masks_true"]
    diam_test = dat["diam_test"] if "diam_test" in dat else 30. * np.ones(
        len(test_noisy))

    imgs_n2s, masks_n2s = [], []

    seg_model = CellposeModel(gpu=True, model_type=f"{ctype}")
    for i in trange(len(test_noisy)):
        out = train_per_image(test_noisy[i])

        masks = seg_model.eval(out, diameter=diam_test[i], channels=[1, 0],
                               channel_axis=0, normalize=True)[0]

        masks_n2s.append(masks)
        imgs_n2s.append(out)

        if plot:
            print(f">>> IMAGE {i}, n_masks = {masks.max()}")
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.imshow(test_noisy[i][0])
            plt.subplot(1, 4, 2)
            plt.imshow(out)
            plt.subplot(1, 4, 3)
            plt.imshow(masks)
            plt.subplot(1, 4, 4)
            plt.imshow(test_labels[i])
            plt.show()

    dat["masks_n2s"] = masks_n2s
    dat["test_n2s"] = imgs_n2s
    if save:
        np.save(root / "noisy_test" / f"test_{noise_type}_n2s.npy", dat)

    return imgs_n2s, masks_n2s


def train_test_specialist(root, n_epochs=50, lr=5e-4, test=True):
    n_train = 3 * 89 * 20
    n_val = 89 * 20

    dat = np.load(root / "noisy_test" / "test_poisson.npy", allow_pickle=True).item()
    test_noisy = dat["test_noisy"][:11]
    masks_true = dat["masks_true"][:11]
    diam_test = dat["diam_test"]

    im_train = [
        io.imread(Path(root / "noisy_test" / "care" / "source" /
                       f"{i:03d}.tif"))[np.newaxis, :, :] for i in range(n_train)
    ]
    im_train.extend(test_noisy)
    im_val = [
        io.imread(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"))
        for i in range(n_train, n_train + n_val)
    ]

    cells_train = Cells(im_train, xy=(128, 128))
    cells_val = Cells(np.array(im_val)[:, np.newaxis, :, :], xy=(128, 128))

    data_loader = DataLoader(cells_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(cells_val, batch_size=64, shuffle=True)

    loss_function = MSELoss()
    masker = Masker(width=4, mode='interpolate')

    model = Unet().to(device)
    tic = time.time()
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(data_loader):
            noisy_images = batch
            net_input, mask = masker.mask(noisy_images, i)
            net_output = model(net_input)
            loss = loss_function(net_output * mask, noisy_images * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += (loss.item()) * len(batch)
        train_loss /= (len(cells_train) / len(cells_val))
        if ep < 10 or ep % 5 == 0 or ep == n_epochs - 1:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    noisy_images = batch
                    net_input, mask = masker.mask(noisy_images, i)
                    net_output = model(net_input)
                    loss = loss_function(net_output * mask, noisy_images * mask)
                    val_loss += (loss.item()) * len(batch)
            #val_loss /= len(cells_val)
            print(
                f"Loss ( {ep} ): \t train: {train_loss:.3f}, val: {val_loss:.3f}, {time.time()-tic:.2f}s"
            )
    if not test:
        return val_loss
    else:

        imgs = []
        masks_n2s = []
        for i in range(len(test_noisy)):
            img = test_noisy[i].copy()
            img, ysub, xsub = transforms.pad_image_ND(img)
            img = torch.from_numpy(img).to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                simple_output = model(img)
            out = simple_output.squeeze()[ysub[0]:ysub[-1] + 1,
                                          xsub[0]:xsub[-1] + 1].cpu().numpy()

            imgs.append(out)
            seg_model = CellposeModel(gpu=True, model_type="cyto2")

            masks = seg_model.eval(out, diameter=diam_test[i], channels=[1, 0],
                                   channel_axis=0, normalize=True)[0]

            masks_n2s.append(masks)

        for i in range(11):
            assert masks_true[i].shape == masks_n2s[i].shape

        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_n2s)
        print(ap.mean(axis=0))

        dat[f"test_n2s"] = imgs
        dat[f"masks_n2s"] = masks_n2s

        np.save(root / "noisy_test" / f"test_poisson_n2s_specialist.npy", dat)

        return imgs, masks_n2s, ap
