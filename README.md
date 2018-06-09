# object_detector

お勉強用(?)に作ってみた物体検知のコード置き場。

## 現在の性能＆他のモデルとの比較メモ

- 訓練データ: PASCAL VOC 07+12 trainval
- 検証データ: PASCAL VOC 07 test
- mAP: mean Average Precision (mAP) with PASCAL VOC2007 metric

|モデル  |入力サイズ|ベースネットワーク|mAP |FPS |
|:-------|:---------|:-----------------|---:|---:|
|SSD321  |321x321   |ResNet101         |77.1|  11|
|SSD300* |300x300   |VGG               |77.2|  46|
|DSOD    |300x300   |DS/64-192-48-1    |77.7|  17|
|DSSD 321|321x321   |ResNet101         |78.6|  14|
|**現在**|320x320   |Darknet53         |79.5|  44|
|YOLOv2  |544x544   |Darknet19         |78.6|  40|
|SSD512* |512x512   |VGG               |79.8|  19|
|YOLOv3  |416x416   |Darknet53         |80.2|    |
|DSSD 513|513x513   |ResNet101         |81.5|   7|
|**現在**|640x640   |Darknet53         |82.8|  22|

FPSはGTX 1080でbatch size 16の処理時間から算出した値なのでだいぶ大き目なはず。(他はたぶん全部Titan X)

## データの配置方法メモ

[DATA.md](./docs/DATA.md)

## モデルの詳細メモ

[MODEL.md](./docs/MODEL.md)
