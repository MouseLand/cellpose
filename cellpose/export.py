"""Auxiliary module for bioimageio format export

Example usage:

```bash
#!/bin/bash

# Define default paths and parameters
DEFAULT_CHANNELS="1 0"
DEFAULT_PATH_PRETRAINED_MODEL="/home/qinyu/models/cp/cellpose_residual_on_style_on_concatenation_off_1135_rest_2023_05_04_23_41_31.252995"
DEFAULT_PATH_README="/home/qinyu/models/cp/README.md"
DEFAULT_LIST_PATH_COVER_IMAGES="/home/qinyu/images/cp/cellpose_raw_and_segmentation.jpg /home/qinyu/images/cp/cellpose_raw_and_probability.jpg /home/qinyu/images/cp/cellpose_raw.jpg"
DEFAULT_MODEL_ID="philosophical-panda"
DEFAULT_MODEL_ICON="🐼"
DEFAULT_MODEL_VERSION="0.1.0"
DEFAULT_MODEL_NAME="My Cool Cellpose"
DEFAULT_MODEL_DOCUMENTATION="A cool Cellpose model trained for my cool dataset."
DEFAULT_MODEL_AUTHORS='[{"name": "Qin Yu", "affiliation": "EMBL", "github_user": "qin-yu", "orcid": "0000-0002-4652-0795"}]'
DEFAULT_MODEL_CITE='[{"text": "For more details of the model itself, see the manuscript", "doi": "10.1242/dev.202800", "url": null}]'
DEFAULT_MODEL_TAGS="cellpose 3d 2d"
DEFAULT_MODEL_LICENSE="MIT"
DEFAULT_MODEL_REPO="https://github.com/kreshuklab/go-nuclear"

# Run the Python script with default parameters
python export.py \
    --channels $DEFAULT_CHANNELS \
    --path_pretrained_model "$DEFAULT_PATH_PRETRAINED_MODEL" \
    --path_readme "$DEFAULT_PATH_README" \
    --list_path_cover_images $DEFAULT_LIST_PATH_COVER_IMAGES \
    --model_version "$DEFAULT_MODEL_VERSION" \
    --model_name "$DEFAULT_MODEL_NAME" \
    --model_documentation "$DEFAULT_MODEL_DOCUMENTATION" \
    --model_authors "$DEFAULT_MODEL_AUTHORS" \
    --model_cite "$DEFAULT_MODEL_CITE" \
    --model_tags $DEFAULT_MODEL_TAGS \
    --model_license "$DEFAULT_MODEL_LICENSE" \
    --model_repo "$DEFAULT_MODEL_REPO"
```
"""

import os
import sys
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse

import torch
import numpy as np

from cellpose.io import imread
from cellpose.utils import download_url_to_file
from cellpose.transforms import pad_image_ND, normalize_img, convert_image
from cellpose.vit_sam import CPnetBioImageIO

from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    Author,
    AxisId,
    ChannelAxis,
    CiteEntry,
    Doi,
    FileDescr,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    LicenseId,
    ModelDescr,
    ModelId,
    OrcidId,
    OutputTensorDescr,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    TorchscriptWeightsDescr,
    Version,
    WeightsDescr,
)
# Define ARBITRARY_SIZE if it is not available in the module
try:
    from bioimageio.spec.model.v0_5 import ARBITRARY_SIZE
except ImportError:
    ARBITRARY_SIZE = ParameterizedSize(min=1, step=1)

from bioimageio.spec.common import HttpUrl
from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model

DEFAULT_CHANNELS = [2, 1]
DEFAULT_NORMALIZE_PARAMS = {
    "axis": -1,
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": False,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False,
}
IMAGE_URL = "http://www.cellpose.org/static/data/rgb_3D.tif"


def download_and_normalize_image(path_dir_temp, channels=DEFAULT_CHANNELS):
    """
    Download and normalize image.
    """
    filename = os.path.basename(urlparse(IMAGE_URL).path)
    path_image = path_dir_temp / filename
    if not path_image.exists():
        sys.stderr.write(f'Downloading: "{IMAGE_URL}" to {path_image}\n')
        download_url_to_file(IMAGE_URL, path_image)
    img = imread(path_image).astype(np.float32)
    img = convert_image(img, channels, channel_axis=1, z_axis=0, do_3D=False, nchan=2)
    img = normalize_img(img, **DEFAULT_NORMALIZE_PARAMS)
    img = np.transpose(img, (0, 3, 1, 2))
    img, _, _ = pad_image_ND(img)
    return img


def load_bioimageio_cpnet_model(path_model_weight, nchan=2):
    cpnet_kwargs = {
        "nout": 3,
    }
    cpnet_biio = CPnetBioImageIO(**cpnet_kwargs)
    state_dict_cuda = torch.load(path_model_weight, map_location=torch.device("cpu"), weights_only=True)
    cpnet_biio.load_state_dict(state_dict_cuda)
    cpnet_biio.eval()  # crucial for the prediction results
    return cpnet_biio, cpnet_kwargs


def descr_gen_input(path_test_input, nchan=2):
    input_axes = [
        SpaceInputAxis(id=AxisId("z"), size=ARBITRARY_SIZE),
        ChannelAxis(channel_names=[Identifier(f"c{i+1}") for i in range(nchan)]),
        SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=16, step=16)),
        SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=16, step=16)),
    ]
    data_descr = IntervalOrRatioDataDescr(type="float32")
    path_test_input = Path(path_test_input)
    descr_input = InputTensorDescr(
        id=TensorId("raw"),
        axes=input_axes,
        test_tensor=FileDescr(source=path_test_input),
        data=data_descr,
    )
    return descr_input


def descr_gen_output_flow(path_test_output):
    output_axes_output_tensor = [
        SpaceOutputAxis(id=AxisId("z"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("z"))),
        ChannelAxis(channel_names=[Identifier("flow1"), Identifier("flow2"), Identifier("flow3")]),
        SpaceOutputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y"))),
        SpaceOutputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x"))),
    ]
    path_test_output = Path(path_test_output)
    descr_output = OutputTensorDescr(
        id=TensorId("flow"),
        axes=output_axes_output_tensor,
        test_tensor=FileDescr(source=path_test_output),
    )
    return descr_output


def descr_gen_output_downsampled(path_dir_temp, nbase=None):
    if nbase is None:
        nbase = [32, 64, 128, 256]

    output_axes_downsampled_tensors = [
        [
            SpaceOutputAxis(id=AxisId("z"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("z"))),
            ChannelAxis(channel_names=[Identifier(f"feature{i+1}") for i in range(base)]),
            SpaceOutputAxis(
                id=AxisId("y"),
                size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("y")),
                scale=2**offset,
            ),
            SpaceOutputAxis(
                id=AxisId("x"),
                size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("x")),
                scale=2**offset,
            ),
        ]
        for offset, base in enumerate(nbase)
    ]
    path_downsampled_tensors = [
        Path(path_dir_temp / f"test_downsampled_{i}.npy") for i in range(len(output_axes_downsampled_tensors))
    ]
    descr_output_downsampled_tensors = [
        OutputTensorDescr(
            id=TensorId(f"downsampled_{i}"),
            axes=axes,
            test_tensor=FileDescr(source=path),
        )
        for i, (axes, path) in enumerate(zip(output_axes_downsampled_tensors, path_downsampled_tensors))
    ]
    return descr_output_downsampled_tensors


def descr_gen_output_style(path_test_style, nchannel=256):
    output_axes_style_tensor = [
        SpaceOutputAxis(id=AxisId("z"), size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId("z"))),
        ChannelAxis(channel_names=[Identifier(f"feature{i+1}") for i in range(nchannel)]),
    ]
    path_style_tensor = Path(path_test_style)
    descr_output_style_tensor = OutputTensorDescr(
        id=TensorId("style"),
        axes=output_axes_style_tensor,
        test_tensor=FileDescr(source=path_style_tensor),
    )
    return descr_output_style_tensor


def descr_gen_arch(cpnet_kwargs, path_cpnet_wrapper=None):
    if path_cpnet_wrapper is None:
        path_cpnet_wrapper = Path(__file__).parent / "resnet_torch.py"
    pytorch_architecture = ArchitectureFromFileDescr(
        callable=Identifier("CPnetBioImageIO"),
        source=Path(path_cpnet_wrapper),
        kwargs=cpnet_kwargs,
    )
    return pytorch_architecture


def descr_gen_documentation(path_doc, markdown_text):
    with open(path_doc, "w") as f:
        f.write(markdown_text)


def package_to_bioimageio(
    path_pretrained_model,
    path_save_trace,
    path_readme,
    list_path_cover_images,
    descr_input,
    descr_output,
    descr_output_downsampled_tensors,
    descr_output_style_tensor,
    pytorch_version,
    pytorch_architecture,
    model_id,
    model_icon,
    model_version,
    model_name,
    model_documentation,
    model_authors,
    model_cite,
    model_tags,
    model_license,
    model_repo,
):
    """Package model description to BioImage.IO format."""
    my_model_descr = ModelDescr(
        id=ModelId(model_id) if model_id is not None else None,
        id_emoji=model_icon,
        version=Version(model_version),
        name=model_name,
        description=model_documentation,
        authors=[
            Author(
                name=author["name"],
                affiliation=author["affiliation"],
                github_user=author["github_user"],
                orcid=OrcidId(author["orcid"]),
            )
            for author in model_authors
        ],
        cite=[CiteEntry(text=cite["text"], doi=Doi(cite["doi"]), url=cite["url"]) for cite in model_cite],
        covers=[Path(img) for img in list_path_cover_images],
        license=LicenseId(model_license),
        tags=model_tags,
        documentation=Path(path_readme),
        git_repo=HttpUrl(model_repo),
        inputs=[descr_input],
        outputs=[descr_output, descr_output_style_tensor] + descr_output_downsampled_tensors,
        weights=WeightsDescr(
            pytorch_state_dict=PytorchStateDictWeightsDescr(
                source=Path(path_pretrained_model),
                architecture=pytorch_architecture,
                pytorch_version=pytorch_version,
            ),
            torchscript=TorchscriptWeightsDescr(
                source=Path(path_save_trace),
                pytorch_version=pytorch_version,
                parent="pytorch_state_dict",  # these weights were converted from the pytorch_state_dict weights.
            ),
        ),
    )

    return my_model_descr


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="BioImage.IO model packaging for Cellpose")
    parser.add_argument("--channels", nargs=2, default=[2, 1], type=int, help="Cyto-only = [2, 0], Cyto + Nuclei = [2, 1], Nuclei-only = [1, 0]")
    parser.add_argument("--path_pretrained_model", required=True, type=str, help="Path to pretrained model file, e.g., cellpose_residual_on_style_on_concatenation_off_1135_rest_2023_05_04_23_41_31.252995")
    parser.add_argument("--path_readme", required=True, type=str, help="Path to README file")
    parser.add_argument("--list_path_cover_images", nargs='+', required=True, type=str, help="List of paths to cover images")
    parser.add_argument("--model_id", type=str, help="Model ID, provide if already exists", default=None)
    parser.add_argument("--model_icon", type=str, help="Model icon, provide if already exists", default=None)
    parser.add_argument("--model_version", required=True, type=str, help="Model version, new model should be 0.1.0")
    parser.add_argument("--model_name", required=True, type=str, help="Model name, e.g., My Cool Cellpose")
    parser.add_argument("--model_documentation", required=True, type=str, help="Model documentation, e.g., A cool Cellpose model trained for my cool dataset.")
    parser.add_argument("--model_authors", required=True, type=str, help="Model authors in JSON format, e.g., '[{\"name\": \"Qin Yu\", \"affiliation\": \"EMBL\", \"github_user\": \"qin-yu\", \"orcid\": \"0000-0002-4652-0795\"}]'")
    parser.add_argument("--model_cite", required=True, type=str, help="Model citation in JSON format, e.g., '[{\"text\": \"For more details of the model itself, see the manuscript\", \"doi\": \"10.1242/dev.202800\", \"url\": null}]'")
    parser.add_argument("--model_tags", nargs='+', required=True, type=str, help="Model tags, e.g., cellpose 3d 2d")
    parser.add_argument("--model_license", required=True, type=str, help="Model license, e.g., MIT")
    parser.add_argument("--model_repo", required=True, type=str, help="Model repository URL")
    return parser.parse_args()
    # fmt: on


def main():
    args = parse_args()

    # Parse user-provided paths and arguments
    channels = args.channels
    model_cite = json.loads(args.model_cite)
    model_authors = json.loads(args.model_authors)

    path_readme = Path(args.path_readme)
    path_pretrained_model = Path(args.path_pretrained_model)
    list_path_cover_images = [Path(path_image) for path_image in args.list_path_cover_images]

    # Auto-generated paths
    path_cpnet_wrapper = Path(__file__).resolve().parent / "resnet_torch.py"
    path_dir_temp = Path(__file__).resolve().parent.parent / "models" / path_pretrained_model.stem
    path_dir_temp.mkdir(parents=True, exist_ok=True)

    path_save_trace = path_dir_temp / "cp_traced.pt"
    path_test_input = path_dir_temp / "test_input.npy"
    path_test_output = path_dir_temp / "test_output.npy"
    path_test_style = path_dir_temp / "test_style.npy"
    path_bioimageio_package = path_dir_temp / "cellpose_model.zip"

    # Download test input image
    img_np = download_and_normalize_image(path_dir_temp, channels=channels)
    np.save(path_test_input, img_np)
    img = torch.tensor(img_np).float()

    # Load model
    cpnet_biio, cpnet_kwargs = load_bioimageio_cpnet_model(path_pretrained_model)

    # Test model and save output
    tuple_output_tensor = cpnet_biio(img)
    np.save(path_test_output, tuple_output_tensor[0].detach().numpy())
    np.save(path_test_style, tuple_output_tensor[1].detach().numpy())
    for i, t in enumerate(tuple_output_tensor[2:]):
        np.save(path_dir_temp / f"test_downsampled_{i}.npy", t.detach().numpy())

    # Save traced model
    model_traced = torch.jit.trace(cpnet_biio, img)
    model_traced.save(path_save_trace)

    # Generate model description
    descr_input = descr_gen_input(path_test_input)
    descr_output = descr_gen_output_flow(path_test_output)
    descr_output_downsampled_tensors = descr_gen_output_downsampled(path_dir_temp, nbase=cpnet_biio.nbase[1:])
    descr_output_style_tensor = descr_gen_output_style(path_test_style, cpnet_biio.nbase[-1])
    pytorch_version = Version(torch.__version__)
    pytorch_architecture = descr_gen_arch(cpnet_kwargs, path_cpnet_wrapper)

    # Package model
    my_model_descr = package_to_bioimageio(
        path_pretrained_model,
        path_save_trace,
        path_readme,
        list_path_cover_images,
        descr_input,
        descr_output,
        descr_output_downsampled_tensors,
        descr_output_style_tensor,
        pytorch_version,
        pytorch_architecture,
        args.model_id,
        args.model_icon,
        args.model_version,
        args.model_name,
        args.model_documentation,
        model_authors,
        model_cite,
        args.model_tags,
        args.model_license,
        args.model_repo,
    )

    # Test model
    summary = test_model(my_model_descr, weight_format="pytorch_state_dict")
    summary.display()
    summary = test_model(my_model_descr, weight_format="torchscript")
    summary.display()

    # Save BioImage.IO package
    package_path = save_bioimageio_package(my_model_descr, output_path=Path(path_bioimageio_package))
    print("package path:", package_path)


if __name__ == "__main__":
    main()
