import functools
import operator
import numpy as np
import dask
import dask.array as da
import dask.delayed as delayed
import ClusterWrap
import time
from cellpose import models


def distributed_eval(
    image,
    blocksize,
    mask=None,
    preprocessing_steps=[],
    model_kwargs={},
    eval_kwargs={},
    cluster_kwargs={},
):
    """
    """

    # set eval defaults
    if 'diameter' not in eval_kwargs.keys():
        eval_kwargs['diameter'] = 30

    # compute overlap
    overlap = eval_kwargs['diameter'] * 2

    # compute mask to array ratio
    if mask is not None:
        ratio = np.array(mask.shape) / image.shape

    # pipeline to run on each block
    def preprocess_and_segment(block, mask=None, block_info=None):

        # get block origin
        origin = np.array(block_info[0]['chunk-location'])
        origin = origin * blocksize - overlap

        # check mask foreground
        if mask is not None:
            mo = np.round(origin * ratio).astype(np.uint16)
            mo = np.maximum(0, mo)
            ms = np.round(blocksize * ratio).astype(np.uint16)
            mask_block = mask[mo[0]:mo[0]+ms[0],
                              mo[1]:mo[1]+ms[1],
                              mo[2]:mo[2]+ms[2],]

            # if there is no foreground, return null result
            if np.sum(mask_block) < 1:
                return np.zeros(block.shape, dtype=np.int64)

        # run preprocessing steps
        image = np.copy(block)
        for pp_step in preprocessing_steps:
            image = pp_step[0](image, **pp_step[1])

        # segment
        model = models.Cellpose(**model_kwargs)
        return model.eval(image, **eval_kwargs)[0]

    # start cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # wrap dataset as a dask object
        if isinstance(image, np.ndarray):
            future = cluster.client.scatter(image)
            image_da = da.from_delayed(
                future, shape=image.shape, dtype=image.dtype,
            )
            image_da = image_da.rechunk(blocksize)
            image_da.persist()
            time.sleep(30)  ### a little time for workers to be allocated
            cluster.client.rebalance()
    
        # a full dataset as a zarr array
        else:
            image_da = da.from_array(image, chunks=blocksize)

        # wrap mask
        mask_d = delayed(mask) if mask is not None else None

        # distribute
        segmentation = da.map_overlap(
            preprocess_and_segment, image_da,
            mask=mask_d,
            depth=overlap,
            dtype=np.int64,
            boundary=0,
            trim=False,
            chunks=[x+2*overlap for x in blocksize],
        )

        # create container for and iterator over blocks
        updated_blocks = np.empty(segmentation.numblocks, dtype=object)
        block_iter = zip(
            np.ndindex(*segmentation.numblocks),
            map(functools.partial(operator.getitem, segmentation),
                da.core.slices_from_chunks(segmentation.chunks))
        )

        # convert local labels to unique global labels
        index, block = next(block_iter)
        updated_blocks[index] = block
        total = da.max(block)
        for index, block in block_iter:
            local_max = da.max(block)
            block += da.where(block > 0, total, 0)
            updated_blocks[index] = block
            total += local_max

        # put blocks back together as dask array
        updated_blocks = da.block(updated_blocks.tolist())

        # TODO: STITCH!

        # TODO: RESULT SHOULD BE WRITTEN TO ZARR
        #    OR RETURN DASK ARRAY AND HAVE AN EXECUTE FUNCTION
        #    WITH COMPUTE OR TO_ZARR OPTIONS

        # return result
        return updated_blocks.compute()
        
