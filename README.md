# object_detector

物体検出の実験用コード置き場。

## 現在の性能

|Model   |backbone      |train    |test  |mAP |備考|
|:-------|:-------------|:--------|:-----|---:|:---|
|`現在`  |ResNet50      |VOC 07+12|VOC 07|74.6||
|YOLOv2  |Darknet19     |VOC 07+12|VOC 07|73.7|FPSを揃えないとフェアじゃないかも|
|SSD300  |VGG           |VOC 07+12|VOC 07|76.8||
|SSD321  |ResNet101     |VOC 07+12|VOC 07|77.1||
|SSD300* |VGG           |VOC 07+12|VOC 07|77.5||
|DSOD300 |DS/64-192-48-1|VOC 07+12|VOC 07|77.7||
|DSSD321 |ResNet101     |VOC 07+12|VOC 07|78.6||

やってることの割になんか低いので、どこかバグってるか変なパラメータがありそう…

あと計算方法によって何故か結構違う(不安):

```txt
mAP=0.7680 mAP(VOC2007)=0.7458
```


## やってること

良さそうなものを手当たり次第にという感じ。

### ネットワーク

ImageNet学習済みResNet50 + FPN風

参考にしたもの: DSSD、DSOD、FPN

あと補助的に、入力からAveragePooling＋重み共有の小さいネットワークをtop-down部分に合流させてみている。(怪)

### 活性化関数

ELU (ほぼ趣味)

### 初期化

最後以外はhe_uniform、最後はzeros。 (ほぼ趣味)

最後をゼロにしておくと学習開始直後がちょっと安定する気がする。

### Prior box

feature map毎に8種類のprior boxを作る。

KMeansを使ってfeature mapに対する相対width/heightの代表値を取得して使用。(使い方は違うが、KMeansを使うのはinspired by YOLOv2)

feature mapは、40x40 ～ 5x5の4つ。(最後だけ奇数にして残りは2倍ずつ)

### 損失関数

Focal loss + L1-smooth loss + Binary crossentropy。

最後のは、予測結果のboxと答えのboxのIoUを回帰して分類のconfidenceと合わせて使用するもの。 (ほぼ趣味)

### DataAugmentation

Random Erasingほか手当たり次第に。

TODO: random crop
