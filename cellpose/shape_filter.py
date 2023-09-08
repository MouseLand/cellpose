import numpy as np
from ClusterWrap.decorator import cluster


def fit_ellipse(mask):

    # read segment as binary mask, generate coordinate array
    coords = np.stack(np.mgrid[tuple(slice(x) for x in mask.shape)].astype(np.float32), axis=-1)

    # get first and second moments
    mean = np.mean(coords[mask], axis=0)
    cov = np.cov(coords[mask], rowvar=False)

    # determine prob threshold to get gaussian ellipse of equal volume to mask
    radii = np.sqrt(np.linalg.eigvalsh(cov))
    norm = (2 * np.pi)**1.5 * np.prod(radii**2)**0.5
    n_stds = np.cbrt( np.sum(mask) / ( 4 * np.pi * np.prod(radii) / 3 ) )
    threshold = np.exp(-0.5 * n_stds**2) / norm

    # create gaussian ellipse at mean, oriented with cov, and equal volume to mask
    diff = coords - mean
    exp = np.einsum('...i,i...', diff, np.einsum('ij,xyzj', np.linalg.inv(cov), diff))
    return ( (np.exp( -0.5 * exp ) / norm) >= threshold )


def mask_ellipse_iou(mask):

    ellipse = fit_ellipse(mask)
    return np.sum( mask * ellipse ) / np.sum( mask + ellipse )


@cluster
def distributed_mask_ellipse_iou(
    masks,
    boxes,
    batch_size,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # batch start indices and function to run on each batch
    start_indices = list(range(0, len(boxes), batch_size))
    box_lists = [boxes[si:min(si+batch_size, len(boxes))] for si in start_indices]
    def wrap_mask_ellipse_iou(start_index, box_list):
        scores = []
        for iii, box in enumerate(box_list):
            index = start_index + iii + 1
            mask = (masks[box] == index)
            scores.append(mask_ellipse_iou(mask))
        return scores

    # map all batches, reformat to a single list
    futures = cluster.client.map(wrap_mask_ellipse_iou, start_indices, box_lists)
    results = cluster.client.gather(futures)
    return [aa for a in results for aa in a]

