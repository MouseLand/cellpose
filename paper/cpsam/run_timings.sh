#!/bin/bash
# 1  2  4  8 16 32 64
for ntiles in 1  2  4  8 16 32 64 128; do
   CMD="mprof run -o ramlogs/mem_2d_${ntiles}_\"$1\"_\"$2\".out timings.py --ntiles ${ntiles} --alg \"$1\" > ramlogs/log_2d_${ntiles}_\"$1\"_\"$2\".out"
   echo "$CMD"
   eval "$CMD"
done

for ntiles in 1  2  4  6 8; do
    CMD="mprof run -o ramlogs/mem_3d_${ntiles}_\"$1\"_\"$2\".out timings.py --use_3D --ntiles ${ntiles} --alg \"$1\" > ramlogs/log_3d_${ntiles}_\"$1\"_\"$2\".out"
    echo "$CMD"
    eval "$CMD"
done

CMD="mprof run -o ramlogs/mem_train_\"$1\"_\"$2\".out timings.py --train --alg \"$1\" > ramlogs/log_train_\"$1\"_\"$2\".out"
echo "$CMD"
eval "$CMD"
