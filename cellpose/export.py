"""Auxiliary module for bioimageio format export"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import torch
import numpy as np

from cellpose.io import imread
from cellpose.utils import download_url_to_file
from cellpose.transforms import pad_image_ND, normalize_img, convert_image
from cellpose.resnet_torch import CPnetBioImageIO

from bioimageio.spec.model.v0_5 import (
    ARBITRARY_SIZE,
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
from bioimageio.spec.common import HttpUrl
from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model


def download_and_normalize_image(path_dir_temp, channels=None):
    if channels is None:
        channels = [2, 1]
    normalize_default = {
        "axis": -1,
        "lowhigh": None,
        "percentile": None,
        "normalize": True,
        "norm3D": False,
        "sharpen_radius": 0,
        "smooth_radius": 0,
        "tile_norm_blocksize": 0,
        "tile_norm_smooth3D": 1,
        "invert": False
    }
    image_url = "http://www.cellpose.org/static/data/rgb_3D.tif"
    filename = os.path.basename(urlparse(image_url).path)
    path_image = path_dir_temp / filename
    if not os.path.exists(path_image):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(image_url, path_image))
        download_url_to_file(image_url, path_image)
    img = imread(path_image).astype(np.float32)
    img = convert_image(img, channels, channel_axis=1, z_axis=0, do_3D=False, nchan=2)
    img = normalize_img(img, **normalize_default)
    img = np.transpose(img, (0, 3, 1, 2))
    img, _, _ = pad_image_ND(img)
    return img


def load_bioimageio_cpnet_model(path_model_weight, nchan=2):
    cpnet_kwargs = {
        "nbase": [nchan, 32, 64, 128, 256],
        "nout": 3,
        "sz": 3,
        "mkldnn": False,
        "conv_3D": False,
        "max_pool": True,
    }
    cpnet_biio = CPnetBioImageIO(**cpnet_kwargs)
    state_dict_cuda = torch.load(path_model_weight, map_location=torch.device("cpu"))
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
    my_model_descr = ModelDescr(
        id=ModelId(model_id),
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


if __name__ == "__main__":
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env is None:
        print("No conda environment found")
    else:
        print(f"Conda environment: {env}")

    # User input paths
    path_dir_model = Path(
        "/g/kreshuk/yu/repositories/bi-unet/resources/configs/2024-athul-ovules/cellpose/contribute_to_cellpose/temp_model_and_input/"
    )
    path_dir_cover_images = path_dir_model
    channels = [1, 0]

    # Auto-generated paths
    path_pretrained_model = path_dir_model / "cp_state_dict_1135_gold.pth"
    list_path_cover_images = [
        path_dir_cover_images / "cellpose_raw_and_segmentation.jpg",
        path_dir_cover_images / "cellpose_raw_and_probability.jpg",
        path_dir_cover_images / "cellpose_raw.jpg",
    ]
    path_cpnet_wrapper = Path(__file__).resolve().parent / "resnet_torch.py"
    path_dir_temp = Path(__file__).resolve().parent.parent / "models" / path_pretrained_model.stem
    Path(path_dir_temp).mkdir(parents=True, exist_ok=True)
    path_save_trace = path_dir_temp / "cp_traced_1135_gold.pt"
    path_test_input = path_dir_temp / "test_input.npy"
    path_test_output = path_dir_temp / "test_output.npy"
    path_test_style = path_dir_temp / "test_style.npy"
    path_readme = path_dir_temp / "README.md"
    path_bioimageio_package = path_dir_temp / "cellpose_gold_1135.zip"

    # Download test input image
    img_np = download_and_normalize_image(path_dir_temp, channels=channels)
    np.save(path_test_input, img_np)
    img = torch.tensor(img_np).float()

    # Load model
    cpnet_biio, cpnet_kwargs = load_bioimageio_cpnet_model(path_pretrained_model)

    # Test model and save ouptut
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
    model_tags_default = ["cellpose", "3d", "2d"]

    # User input arguments
    readme = """# A User-trained Cellpose Model

A Cellpose nuclei model fine-tuned on nuclei data for testing purposes.
"""
    descr_gen_documentation(path_readme, readme)
    model_id = "philosophical-panda"
    model_icon = "🐼"
    model_version = "0.1.0"
    model_name = "Cellpose Plant Nuclei ResNet"
    model_documentation = "An experimental Cellpose nuclear model fine-tuned on ovules 1136, 1137, 1139, 1170 and tested on ovules 1135 (see reference for dataset details). A model for BioImage.IO team to test and develop post-processing tools."
    model_authors = [
        {"name": "Qin Yu", "affiliation": "EMBL", "github_user": "qin-yu", "orcid": "0000-0002-4652-0795"},
    ]
    model_cite = [
        {
            "text": "For more details of the model itself, see the manuscript",
            "doi": "10.1101/2024.02.19.580954",
            "url": None,
        },
    ]
    model_tags = model_tags_default + ["nuclei"]
    model_license = "MIT"
    model_repo = "https://github.com/kreshuklab/go-nuclear"

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
    )

    # Test model
    summary = test_model(my_model_descr, weight_format="pytorch_state_dict")
    summary.display()
    summary = test_model(my_model_descr, weight_format="torchscript")
    summary.display()

    print("package path:", save_bioimageio_package(my_model_descr, output_path=Path(path_bioimageio_package)))
