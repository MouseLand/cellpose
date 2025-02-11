Distributed Cellpose for Larger-Than-Memory Data
------------------------------------------------

The `cellpose.contrib.distributed_cellpose` module is intended to help run cellpose on datasets
that are too large to fit in system memory. The dataset is divided into overlapping blocks and
each block is segmented separately. Results are stitched back together into a seamless segmentation
of the whole dataset.

Blocks can be run in parallel, in series, or both. Compute resources (GPUs, CPUs, and RAM) can be
arbitrarily partitioned. 
