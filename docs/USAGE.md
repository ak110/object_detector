# 使い方メモ

コマンドライン例。

## PASCAL VOC 2007+2012 trainval

    mpirun -np 4 -H localhost:4 ./voc_train.py
    ./voc_evaluate.py
    ./voc_validate.py

## base-network / input-size / map-sizes例

    --input-size=512 --map-sizes 64 32 16
    --base-network=custom --input-size=773 --map-sizes 48 24 12 --batch-size=8
    --base-network=custom --input-size=1023 --map-sizes 64 32 16 --batch-size=8
