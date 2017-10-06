# object_detector

物体検出の実験用コード置き場。

## 現在の性能

VOC07+12 trainvalで学習してVOC07 testで検証。

```txt
601s - loss: 1.9327 - loss_loc: 1.2565 - acc_bg: 1.0000 - acc_obj: 0.3237 - val_loss: 1.9300 - val_loss_loc: 1.2857 - val_acc_bg: 0.9999 - val_acc_obj: 0.3535
mAP=0.7544 mAP(VOC2007)=0.7329
```

(やってることの割になんか低いので、どこかバグってるか変なパラメータがありそう…)

## やってること

良さそうなものを手当たり次第にという感じ。

### ネットワーク

ImageNet学習済みResNet50 + FPN風

参考: DSSD、DSOD、FPN

あと補助的に小さいネットワークで、入力からAveragePooling＋重み共有のCNNの結果をFPN風のtop-down部分に合流させてみている。(怪)

### 活性化関数

ELU (怪: ほぼ趣味)

### Prior box

feature map毎に8種類のprior boxを作る。

KMeansを使ってfeature mapに対する相対width/heightの代表値を取得して使用。(使い方は違うが、KMeansを使うのはinspired by YOLOv2)

feature mapは、40x40 ～ 5x5の4つ。(最後だけ奇数にして残りは2倍ずつ)

### 損失関数

Focal loss + L1-smooth loss。

### DataAugmentation

Random Erasingほか手当たり次第に。

random cropはせず、代わりに上下左右にランダムサイズのパディング。(とりあえず)
