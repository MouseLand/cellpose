"""Tests for distributed_segmentation.py module."""
import pathlib

import dask
import dask.array as da
import numpy as np
import pytest
import tifffile
from dask.array.utils import assert_eq

from cellpose.contrib import distributed_segmentation

TESTDATA = pathlib.Path(__file__).parent / "testdata"

def temp_test_segment_nucleus(data_dir):
    """Tests segmentation on dask array."""
    tiff_input = data_dir.joinpath('distributed').joinpath("segment_80x224x448_input.tiff")
    tiff_expected = data_dir.joinpath('distributed').joinpath("segment_80x224x448_expected.tiff")
    image = tifffile.imread(tiff_input) #TESTDATA / "segment_80x224x448_input.tiff")
    image = image[..., None]
    image = da.from_array(image).rechunk({2: 128})

    actual = distributed_segmentation.segment(
        image,
        [0, 0],
        "cyto",
        diameter=(10, 30, 30),
        fast_mode=True,
        use_anisotropy=False,
    )

    # Assume segmenting on a single GPU and process each chunk serially.
    with dask.config.set(scheduler="synchronous"):
        actual = actual.compute()

    expected = tifffile.imread(tiff_expected)#TESTDATA / "segment_80x224x448_expected.tiff")
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        ([0, 1, 1, 2, 2, 0], [0, 1, 1, 1, 1, 0]),
        (
            [
                [0, 1, 1, 2, 2, 0],
                [0, 1, 1, 2, 2, 0],
            ],
            [
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
            ],
        ),
        (
            [
                [0, 1, 1, 2, 2, 0],
                [0, 3, 3, 4, 4, 0],
            ],
            [
                [0, 1, 1, 1, 1, 0],
                [0, 2, 2, 2, 2, 0],
            ],
        ),
        (
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 2, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ),
        (
            [
                [0, 1, 1, 2, 2, 0],
                [0, 3, 3, 4, 4, 0],
                [0, 3, 3, 4, 4, 0],
                [0, 5, 5, 6, 6, 0],
                [0, 5, 5, 6, 6, 0],
                [0, 7, 7, 8, 8, 0],
            ],
            [
                [0, 1, 1, 1, 1, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 3, 3, 3, 3, 0],
            ],
        ),
        (
            [
                [1, 2, 3, 7, 8, 9],
                [4, 5, 5, 10, 10, 11],
                [6, 5, 5, 10, 10, 12],
                [13, 14, 14, 19, 19, 20],
                [15, 14, 14, 19, 19, 21],
                [16, 17, 18, 22, 23, 24],
            ],
            [
                [1, 2, 3, 2, 3, 7],
                [4, 5, 5, 5, 5, 8],
                [6, 5, 5, 5, 5, 9],
                [4, 5, 5, 5, 5, 8],
                [6, 5, 5, 5, 5, 9],
                [10, 11, 12, 11, 12, 13],
            ],
        ),
    ],
)
def test_link_labels(data, expected):
    data = np.array(data, dtype=np.int32)
    nlabels = np.max(data)
    data = da.asarray(data, chunks=3)
    result = distributed_segmentation.link_labels(data, nlabels, depth=1)
    assert_eq(result, np.array(expected, dtype=np.int32))


def test_link_labels_complex_case():
    """Case with asymmetric depth.

    Core is 4x6:
    111110
    111110
    111022
    011033
    """
    data = [
        [0, 0, 0, 1, 1, 0, 0, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 2, 3, 3, 0, 4, 4, 4, 4],
        [0, 0, 5, 5, 5, 5, 5, 9, 9, 9, 9, 0, 0, 11],
        [0, 0, 5, 5, 5, 0, 6, 9, 9, 0, 10, 10, 11, 11],
        [0, 0, 0, 5, 5, 0, 7, 9, 9, 0, 11, 11, 11, 11],
        [0, 0, 0, 8, 8, 0, 7, 12, 12, 0, 11, 13, 13, 14],
    ]
    data = np.array(data, dtype=np.int32)
    data = da.asarray(data, chunks=(4, 7))

    expected = [
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 2, 1, 1, 0, 2, 2, 2, 2],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3],
        [0, 0, 1, 1, 1, 0, 2, 1, 1, 0, 2, 2, 3, 3],
        [0, 0, 0, 1, 1, 0, 3, 1, 1, 0, 3, 3, 3, 3],
        [0, 0, 0, 4, 4, 0, 3, 4, 4, 0, 3, 5, 5, 6],
    ]
    result = distributed_segmentation.link_labels(data, np.max(data), depth=(1, 2))
    assert_eq(result, np.array(expected, dtype=np.int32))


def test_link_labels_threshold():
    data = [
        [0, 1, 1, 2, 2, 0],
        [0, 1, 1, 2, 2, 0],
        [0, 1, 1, 3, 3, 0],
    ]
    data = np.array(data, dtype=np.int32)
    data = da.asarray(data, chunks=3)

    expected = [
        [0, 1, 1, 2, 2, 0],
        [0, 1, 1, 2, 2, 0],
        [0, 1, 1, 3, 3, 0],
    ]
    result = distributed_segmentation.link_labels(
        data, np.max(data), depth=1, iou_threshold=0.8
    )
    assert_eq(result, np.array(expected, dtype=np.int32))

    expected = [
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 2, 2, 0],
    ]
    result = distributed_segmentation.link_labels(
        data, np.max(data), depth=1, iou_threshold=0.5
    )
    assert_eq(result, np.array(expected, dtype=np.int32))
