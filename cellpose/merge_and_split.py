import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.ndimage import find_objects
from scipy.ndimage import label as ndi_label
import dask.array as da
import dask_image.ndmeasure._utils._label as label
import zarr
from ClusterWrap.decorator import cluster
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def merge_small_segments(
    masks,
    boxes,
    minimum_box_volume,
):
    """
    """

    # copy boxes, get all volumes, get small ones, initialize remapping
    boxes = [tuple(slice(0, x) for x in masks.shape),] + deepcopy(boxes)
    get_volume = lambda box: np.prod( [a.stop - a.start for a in box] )
    volumes = [get_volume(box) for box in boxes]
    too_small_indices = list(np.nonzero( np.array(volumes) < minimum_box_volume )[0])
    merge_groups = [[x] for x in range(len(boxes))]

    # merge until all boxes are big enough
    while len(too_small_indices) > 0:

        # get data, create binary mask
        label = too_small_indices[0]
        data = masks[tuple(slice(max(0, a.start - 1), a.stop + 1) for a in boxes[label])]
        binary_mask = np.isin(data, merge_groups[label])

        # get neighbors and contact areas
        boundary = np.bitwise_xor(binary_dilation(binary_mask), binary_mask)
        neighbors, areas = np.unique(data[boundary], return_counts=True)
        if neighbors[0] == 0: neighbors, areas = neighbors[1:], areas[1:]

        # find correct merge label
        if len(neighbors) == 0: merge_label = 0
        elif len(neighbors) == 1: merge_label = neighbors[0]
        elif len(neighbors) > 1:
            largest_neighbors = [neighbors[x] for x in np.nonzero( areas == areas.max() )[0]]
            if len(largest_neighbors) == 1: merge_label = largest_neighbors[0]
            else:
                neighbor_volumes = [volumes[x] for x in largest_neighbors]
                merge_label = largest_neighbors[np.argmin(neighbor_volumes)]

        # merge labels, store new box and volume
        merged_group = merge_groups[label] + merge_groups[merge_label]
        merged_box = merge_boxes(boxes[label], boxes[merge_label])
        merged_volume = get_volume(merged_box)
        for member in merged_group:
            merge_groups[member] = merged_group
            boxes[member] = merged_box
            volumes[member] = merged_volume

        # maintain list
        if merged_volume >= minimum_box_volume:
            for member in merged_group:
                try: too_small_indices.remove(member)
                except: pass

    # return new labeling array and boxes
    new_labeling = np.array([np.min(x) for x in merge_groups])
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    boxes = [boxes[x] for x in unique[1:]]  # skip 0
    return new_labeling, boxes


def split_large_segments(
    masks,
    boxes,
    maximum_box_volume,
    average_cell_volume=40000,
    min_distance=10,
):
    """
    """

    # record largest label value, copy boxes, get all volumes, get large ones
    max_label = len(boxes)
    boxes = deepcopy(boxes)
    get_volume = lambda box: np.prod( [a.stop - a.start for a in box] )
    volumes = [get_volume(box) for box in boxes]
    too_large_indices = list(np.nonzero( np.array(volumes) > maximum_box_volume )[0])

    # more efficient to store this now
    footprint = np.ones((3,3,3))
    structure = np.ones((4, 7, 7))

    # split all large segments
    for iii, index in enumerate(too_large_indices):

        # print basic progress
        if iii % 100 == 0: print(iii, ' out of ', len(too_large_indices))

        # get data, create binary mask
        box = tuple(slice(a.start - 1, a.stop + 1) for a in boxes[index])
        data = masks[box]
        foreground = data == index + 1
        binary_mask = foreground.astype(data.dtype)

        # get seeds
        num_peaks = int(round(np.sum(foreground) / average_cell_volume))
        if num_peaks < 2: continue
        binary_mask_eroded = binary_erosion(binary_mask, structure=structure).astype(data.dtype)
        distances = distance_transform_edt(binary_mask_eroded)
        coords = peak_local_max(
            distances,
            num_peaks=num_peaks,
            min_distance=min_distance,
            footprint=footprint,
            labels=binary_mask_eroded,
        )
        markers = np.zeros(binary_mask.shape, dtype=bool)
        markers[tuple(coords.T)] = True
        markers, _ = ndi_label(markers)

        # split
        split = watershed(
            binary_mask,    # cleaner result than when using -distances
            markers=markers,
            mask=binary_mask,
            compactness=0.1,
        )
        n_new_labels = split.max()
        if n_new_labels < 2: continue

        # get new bounding boxes in global coordinates
        new_boxes = find_objects(split)
        move_origin = lambda new_box: tuple(slice(a.start+b.start, a.start+b.stop) for a, b in zip(box, new_box))
        new_boxes = [move_origin(x) for x in new_boxes]
        boxes[index] = new_boxes[0]
        boxes += new_boxes[1:]

        # remap to globally unique values
        split[foreground] += max_label - 1
        split[split == max_label] = index + 1
        max_label += n_new_labels - 1

        # integrate with existing masks and write data
        data[foreground] = split[foreground]
        masks[box] = data

    # return updated masks and boxes
    return masks, boxes

        


@cluster
def relabel_segments(
    masks,
    new_labeling,
    write_path,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    masks_da = da.from_zarr(masks)
    new_labeling_da = da.from_array(new_labeling, chunks=-1)
    relabeled = label.relabel_blocks(masks_da, new_labeling_da)
    da.to_zarr(relabeled, write_path)
    return zarr.open(write_path, mode='r+')


def merge_boxes(box1, box2):
    """
    """

    union = []
    for a, b in zip(box1, box2):
        start = min(a.start, b.start)
        stop = max(a.stop, b.stop)
        union.append( slice(start, stop) )
    return tuple(union)


