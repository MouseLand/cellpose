import io

import numpy as np
import torch
from openvino.inference_engine import IECore

ie = IECore()

def to_openvino(model):
    if isinstance(model.net, OpenVINOModel):
        return model
    model.mkldnn = False
    model.net.mkldnn = False
    model.net = OpenVINOModel(model.net)
    return model


class OpenVINOModel(object):
    def __init__(self, model):
        self._base_model = model
        self._nets = {}
        self._exec_nets = {}
        self._model_id = "default"


    def _init_model(self, inp):
        if self._model_id in self._nets:
            return self._nets[self._model_id], self._exec_nets[self._model_id]

        # Load a new instance of the model with updated weights
        if self._model_id != "default":
            self._base_model.load_model(self._model_id, cpu=True)

        buf = io.BytesIO()
        dummy_input = torch.zeros([1] + list(inp.shape[1:]))  # To avoid extra network reloading we process batch in the loop
        torch.onnx.export(self._base_model, dummy_input, buf, input_names=["input"], output_names=["output", "style"])
        net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
        exec_net = ie.load_network(net, "CPU")

        self._nets[self._model_id] = net
        self._exec_nets[self._model_id] = exec_net

        return net, exec_net


    def __call__(self, inp):
        net, exec_net = self._init_model(inp)

        batch_size = inp.shape[0]
        if batch_size > 1:
            out_shape = net.outputs["output"].shape
            style_shape = net.outputs["style"].shape
            output = np.zeros([batch_size] + out_shape[1:], np.float32)
            style = np.zeros([batch_size] + style_shape[1:], np.float32)
            for i in range(batch_size):
                out = exec_net.infer({"input": inp[i : i + 1]})
                output[i] = out["output"]
                style[i] = out["style"]

            return torch.tensor(output), torch.tensor(style)
        else:
            out = exec_net.infer({"input": inp})
            return torch.tensor(out["output"]), torch.tensor(out["style"])


    def load_model(self, path, cpu):
        self._model_id = path
        return self


    def eval(self):
        pass
