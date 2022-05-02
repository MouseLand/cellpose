import io

import numpy as np
import torch
from openvino.runtime import Core

ie = Core()

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
        self._model_id = "default"


    def _init_model(self, inp):
        if self._model_id in self._nets:
            return self._nets[self._model_id]

        # Load a new instance of the model with updated weights
        if self._model_id != "default":
            self._base_model.load_model(self._model_id, cpu=True)

        buf = io.BytesIO()
        dummy_input = torch.zeros([1] + list(inp.shape[1:]))  # To avoid extra network reloading we process batch in the loop
        torch.onnx.export(self._base_model, dummy_input, buf, input_names=["input"], output_names=["output", "style"])
        net = ie.read_model(buf.getvalue(), b"")
        exec_net = ie.compile_model(net, "CPU").create_infer_request()

        self._nets[self._model_id] = exec_net

        return exec_net


    def __call__(self, inp):
        exec_net = self._init_model(inp)

        batch_size = inp.shape[0]
        if batch_size > 1:
            outputs = []
            styles = []
            for i in range(batch_size):
                outs = exec_net.infer({"input": inp[i : i + 1]})
                outs = {out.get_any_name(): value for out, value in outs.items()}
                outputs.append(outs["output"])
                styles.append(outs["style"])

            outputs = np.concatenate(outputs)
            styles = np.concatenate(styles)
            return torch.tensor(outputs), torch.tensor(styles)
        else:
            outs = exec_net.infer({"input": inp})
            outs = {out.get_any_name(): value for out, value in outs.items()}
            return torch.tensor(outs["output"]), torch.tensor(outs["style"])


    def load_model(self, path, cpu):
        self._model_id = path
        return self


    def eval(self):
        pass
