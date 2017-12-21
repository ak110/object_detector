# 使い方メモ

コマンドライン例。

## PASCAL VOC 2007+2012

```sh
./preprocess.py
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py
./validate.py
```

## custom network: VOC pretrain + CSV

```sh
./preprocess.py --base-network=custom --input-size=773 --map-sizes 48 24 12 6
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py --batch-size=4 --no-lr-decay
./validate.py
mv results/model.h5 results/model.base.h5

./preprocess.py --data-type=csv --base-network=custom --input-size=773 --map-sizes 48 24 12 6
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH ./train.py --batch-size=4 --no-lr-decay
./validate.py
```
