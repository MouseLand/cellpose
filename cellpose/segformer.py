"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import torch
from torch import nn

try:
    import segmentation_models_pytorch as smp

    class Transformer(nn.Module):
        """ Transformer encoder from segformer paper with MAnet decoder 
            (configuration from MEDIAR)
        """

        def __init__(self, encoder="mit_b5", encoder_weights=None, decoder="MAnet",
                     diam_mean=30.):
            super().__init__()
            net_fcn = smp.MAnet if decoder == "MAnet" else smp.FPN
            self.encoder = encoder
            self.decoder = decoder
            self.net = net_fcn(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                # (use "imagenet" pre-trained weights for encoder initialization if training)
                in_channels=3,
                classes=3,
                activation=None)
            self.nout = 3
            self.mkldnn = False
            self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean,
                                          requires_grad=False)
            self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean,
                                            requires_grad=False)

        def forward(self, X):
            # have to convert to 3-chan (RGB)
            if X.shape[1] < 3:
                X = torch.cat(
                    (X,
                     torch.zeros((X.shape[0], 3 - X.shape[1], X.shape[2], X.shape[3]),
                                 device=X.device)), dim=1)
            y = self.net(X)
            return y, torch.zeros((X.shape[0], 256), device=X.device)

        @property
        def device(self):
            return next(self.parameters()).device

        def save_model(self, filename):
            """
            Save the model to a file.

            Args:
                filename (str): The path to the file where the model will be saved.
            """
            torch.save(self.state_dict(), filename)

        def load_model(self, filename, device=None):
            """
            Load the model from a file.

            Args:
                filename (str): The path to the file where the model is saved.
                device (torch.device, optional): The device to load the model on. Defaults to None.
            """
            if (device is not None) and (device.type != "cpu"):
                state_dict = torch.load(filename, map_location=device)
            else:
                self.__init__(encoder=self.encoder, decoder=self.decoder,
                              diam_mean=self.diam_mean)
                state_dict = torch.load(filename, map_location=torch.device("cpu"))

            self.load_state_dict(
                dict([(name, param) for name, param in state_dict.items()]),
                strict=False)

except Exception as e:
    print(e)
    print("need to install segmentation_models_pytorch to run transformer")
