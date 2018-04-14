# 使い方メモ

コマンドライン例。

## PASCAL VOC 2007+2012

    ./preprocess.py
    mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py
    ./validate.py
    ./report.py

## custom network: VOC pretrain + CSV

    ./preprocess.py --data-type=voc --base-network=custom --input-size=773 --map-sizes 48 24 12
    mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py --batch-size=4 --no-lr-decay
    ./validate.py
    ./report.py
    mv results/model.h5 results/model.base.h5

    ./preprocess.py --data-type=csv --base-network=custom --input-size=773 --map-sizes 48 24 12
    mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py --batch-size=4
    ./validate.py
    ./report.py

## base-network / input-size / map-sizes例

    --input-size=512 --map-sizes 64 32 16
    --base-network=custom --input-size=773 --map-sizes 48 24 12
    --base-network=custom --input-size=1023 --map-sizes 64 32 16
