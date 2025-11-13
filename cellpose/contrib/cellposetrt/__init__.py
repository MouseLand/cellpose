"""TensorRT-backed Cellpose model module.

  TensorRT is NVIDIA's neural‑network inference compiler/runtime for NVIDIA GPUs. It takes
  an ONNX/graph, picks optimized kernels, fuses layers, and plans memory/scheduling
  specialized for a given GPU architecture and fixed input profile.

  By specializing for fixed input shapes and fusing ops, TensorRT can deliver
  higher performance than standard PyTorch inference (1.7x speedup in RTX 5090).
  A CellposeSAM model can be converted to the TensorRT format by running
  cellpose/contrib/cellposetrt/trt_build.py.

  `CellposeModelTRT(engine_path=...)` in this module is a drop-in replacement for
  the standard `CellposeModel` to run CellposeSAM via TensorRT.
"""

from pathlib import Path

import tensorrt as trt
import torch

from cellpose import models


class TRTEngineModule(torch.nn.Module):
    """TensorRT-backed CellposeSAM model.

    It is not intended for auxiliary training/export variants that add extra outputs
    (e.g., BioImage.IO downsampled tensors, denoise/perceptual losses).

    Notes
    - Requires TensorRT >= 10.
    - Engines are compiled for fixed profiles batch size and tile size.
    """
    def __init__(self, engine_path: str|Path, device=torch.device("cuda")):
        super().__init__()

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"TensorRT backend requires a CUDA device, got '{self.device.type}'. CPUs/MLX are unsupported."
            )

        ver = getattr(trt, "__version__", None)
        if not ver:
            raise RuntimeError("TensorRT >= 10 required (version unknown).")
        try:
            major = int(str(ver).split(".")[0])
        except Exception:
            raise RuntimeError(f"TensorRT >= 10 required (found {ver}).")
        if major < 10:
            raise RuntimeError(f"TensorRT >= 10 required (found {ver}).")

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            except Exception as exc:
                raise ValueError(f"{engine_path} is not a valid TensorRT engine") from exc

        self._ctx = self._engine.create_execution_context()

        # Names exported by our ONNX: 'input' -> y/style
        self._name_in = "input"
        self._name_y = "y"
        self._name_s = "style"

        def _to_torch_dtype(dt):
            if dt == trt.DataType.BF16:
                return torch.bfloat16
            if dt == trt.DataType.HALF:
                return torch.float16
            if dt == trt.DataType.FLOAT:
                return torch.float32
            raise ValueError(f"Unsupported TensorRT dtype: {dt}")

        # Sanity: make sure the names exist and modes are right
        for name in (self._name_in, self._name_y, self._name_s):
            self._engine.get_tensor_dtype(name)

        # Capture per-tensor dtypes from engine
        self._dtype_in = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_in))
        self._dtype_y = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_y))
        self._dtype_s = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_s))

        self.dtype = self._dtype_in

        # Detect fixed batch dimension from engine input shape (None if dynamic)
        self._in_dims = tuple(self._engine.get_tensor_shape(self._name_in))  # (N,C,H,W) with -1 for dynamic dims
        self._fixedN = self._in_dims[0] if self._in_dims[0]> 0 else None

    def forward(self, X: torch.Tensor):
        if not X.is_cuda:
            raise RuntimeError("Input must be a CUDA tensor")
        if X.device != self.device:
            X = X.to(self.device, non_blocking=True)
        if X.dtype != self._dtype_in:
            X = X.to(self._dtype_in)
        X = X.contiguous()
        N, C, H, W = X.shape

        # Require exact N match when engine has fixed batch.
        if self._fixedN is not None and N != self._fixedN:
            raise ValueError(
                f"Input batch {N} must equal engine fixed batch N={self._fixedN}. "
                f"Adjust batch_size or rebuild the engine."
            )
        effective_N = self._fixedN or N

        # 1) Set input shape by name
        self._ctx.set_input_shape(self._name_in, (effective_N, C, H, W))

        # 2) Allocate outputs; query shapes from engine (may have -1 -> allocate by heuristics if needed)
        #    Cellpose heads are [N,3,H,W] and [N,S], so we can shape from input.
        # Read S from engine if available; otherwise default to 256 (Cellpose style vec size) and adjust if needed.
        try:
            # If engine carries concrete dims (profile-dependent), use them
            s_dims = tuple(self._engine.get_tensor_shape(self._name_s))
            if any(d < 0 for d in s_dims):
                S = s_dims[-1] if s_dims[-1] > 0 else 256
            else:
                S = s_dims[-1]
        except Exception:
            S = 256

        y = torch.empty((effective_N, 3, H, W), device=X.device, dtype=self._dtype_y)
        s = torch.empty((effective_N, S), device=X.device, dtype=self._dtype_s)

        stream = torch.cuda.current_stream(self.device)
        stream_handle = int(stream.cuda_stream)

        self._ctx.set_tensor_address(self._name_in, int(X.data_ptr()))
        self._ctx.set_tensor_address(self._name_y, int(y.data_ptr()))
        self._ctx.set_tensor_address(self._name_s, int(s.data_ptr()))

        ok = self._ctx.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return y, s


class CellposeModelTRT(models.CellposeModel):
    """Drop-in replacement for CellposeModel (eval) using TensorRT.

    Preparation
    - Build an engine for your model first with scripts/trt_build.py, for example:
      python scripts/trt_build.py PRETRAINED -o OUTPUT.plan --batch-size 4 --bsize 256
      Then pass engine_path=OUTPUT.plan to this class.

    Contract
    - Uses a TensorRT engine whose forward returns exactly (y, style) as defined
      in TRTEngineModule; aligns with the main segmentation pipeline.
    - Not intended for denoise/perceptual-loss training utilities or BioImage.IO
      export paths that expect additional tensors beyond (y, style).
    """

    def __init__(
        self,
        gpu=False,
        pretrained_model="cyto2",
        model_type=None,
        diam_mean=None,
        device=None,
        nchan=None,
        use_bfloat16=True,
    ):
        engine_path = pretrained_model
        if engine_path is None:
            raise ValueError("TensorRT engine (.plan) must be generated from `trt_build.py` and provided via `pretrained_model`.")
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found at {engine_path}")
        self.engine_path = engine_path

        super().__init__(
            gpu=gpu,
            pretrained_model="cpsam",  # dummy, not used
            model_type=model_type,
            diam_mean=diam_mean,
            device=device,
            nchan=nchan,
            use_bfloat16=True,
        )
        dev = torch.device("cuda" if device is None else device)
        if not use_bfloat16:
            raise ValueError("CellposeModelTRT only supports use_bfloat16=True")

        self.net = TRTEngineModule(engine_path, device=dev)

    def eval(self, x, **kwargs):
        if kwargs.get("bsize", 256) != self.net._in_dims[2]:
            raise ValueError(f"This engine only supports bsize={self.net._in_dims[2]} (built with this bsize)")
        return super().eval(x, **kwargs)
