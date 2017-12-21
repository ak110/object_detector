# object_detector

お勉強用(?)に作ってみた物体検知のコード置き場。

## 現在の性能＆他のモデルとの比較メモ

- 訓練データ: PASCAL VOC 07+12 trainval
- 検証データ: PASCAL VOC 07 test
- mAP: mean Average Precision (mAP) with PASCAL VOC2007 metric

|モデル  |入力サイズ|ベースネットワーク|mAP |FPS |備考                                    |
|:-------|:---------|:-----------------|---:|---:|:---------------------------------------|
|**現在**|319x319   |ResNet50          |75.6|  19|FPSはGTX 1080。(他はたぶん全部Titan X)  |
|**現在**|512x512   |ResNet50          |77.1|  15|FPSはGTX 1080。(他はたぶん全部Titan X)  |
|SSD321  |321x321   |ResNet101         |77.1|  11|                                        |
|SSD300* |300x300   |VGG               |77.2|  46|                                        |
|DSOD    |300x300   |DS/64-192-48-1    |77.7|  17|                                        |
|DSSD 321|321x321   |ResNet101         |78.6|  14|                                        |
|YOLO9000|544x544   |Darknet19         |78.6|  40|                                        |
|SSD512* |512x512   |VGG               |79.8|  19|                                        |
|DSSD 513|513x513   |ResNet101         |81.5|   7|                                        |

良さそうなものを手当たり次第にやっているつもりな割に、性能はぱっとしない…。

```txt
epoch=70 mAP=0.7795 mAP(VOC2007)=0.7557
epoch=128 mAP=0.7961 mAP(VOC2007)=0.7706
```

## データの配置方法メモ

[DATA.md](./DATA.md)

## 使い方メモ

[USAGE.md](./USAGE.md)

## モデルの詳細メモ

[MODEL.md](./MODEL.md)
