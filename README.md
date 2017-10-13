# object_detector

お勉強用(?)に作ってみた物体検知のコード置き場。

## 現在の性能＆他のモデルとの比較メモ

- 訓練データ: PASCAL VOC 07+12 trainval
- 検証データ: PASCAL VOC 07 test
- mAP: mean Average Precision (mAP) with PASCAL VOC2007 metric

|モデル  |入力サイズ|ベースネットワーク|mAP |FPS |
|:-------|:---------|:-----------------|---:|---:|
|**現在**|319x319   |ResNet50          |74.6|  19|
|SSD321  |321x321   |ResNet101         |77.1|  11|
|SSD300* |300x300   |VGG               |77.2|  46|
|DSOD    |300x300   |DS/64-192-48-1    |77.7|  17|
|DSSD 321|321x321   |ResNet101         |78.6|  14|
|YOLOv2  |544x544   |Darknet19         |78.6|  40|
|SSD512* |512x512   |VGG               |79.8|  19|
|DSSD 513|513x513   |ResNet101         |81.5|   7|

良さそうなものを手当たり次第にやっているつもりな割に、性能はぱっとしない…。

```txt
mAP=0.7669 mAP(VOC2007)=0.7457
```

## やってること

### ネットワーク

ResNet50 + FPN風

参考にしたもの: [DSSD](https://arxiv.org/abs/1701.06659)、[DSOD](https://arxiv.org/abs/1708.01241)、[FPN](https://arxiv.org/abs/1612.03144)など。

### 活性化関数

[ELU](https://arxiv.org/abs/1511.07289) (ほぼ趣味)

### 初期化

最後以外はhe_uniform、最後はzeros。 (ほぼ趣味)

最後をゼロにしておくと学習開始直後がちょっと安定する気がする。

### Prior box

使用するfeature mapは、40x40 ～ 5x5の4つ。(最後だけ奇数にして残りは2倍ずつにしてみた)

feature map毎に8種類のprior boxを作る。

パラメータとして、grid cellのサイズに対するwidth/heightの割合を8セット必要とする。(全feature mapで共有)

例えばその1つを `widthの割合 = 1.5` 、 `heightの割合 = 2.0` とすると、
40x40のfeature mapに対しては、Prior boxの大きさが以下のようになる。

```txt
横幅 = 入力画像の横幅 × 1/40 × 1.5
縦幅 = 入力画像の縦幅 × 1/40 × 2.0
```

このパラメータは、事前に訓練データからKMeansを使って決める。
(YOLOv2のDimension Clustersの簡易版のつもり)

### 損失関数：分類

[Focal loss](https://arxiv.org/abs/1708.02002)

### 損失関数：bounding box

L1-smooth loss

`x, y, w, h` ではなく `x1, y1, x2, y2` でやっている。(怪)

### 損失関数：IoU

binary crossentropy

予測結果のboxと答えのboxのIoUを予測(回帰)して、分類のconfidenceと合わせて使用するもの。
([自分で考案したつもりがYOLOv1が既にやっていた](https://twitter.com/ak11/status/917901136782278656)。)

合わせ方は加算（相加平均）、乗算（相乗平均）、調和平均とか試したけど乗算が僅差で良かったので採用。

### DataAugmentation

[Random Erasing](https://arxiv.org/abs/1708.04896)ほか手当たり次第に。

Random Erasingは、あまりbounding boxを隠しすぎないように色々手動で条件を付けてみている。

## 作ってて困ったポイント備忘録

困ったら[SSDのKeras実装](https://github.com/rykov8/ssd_keras)と[ChainerCV](https://github.com/chainer/chainercv)を見たら大体解決した。ありがたい。

### mAP

計算方法の情報が見つからなくて苦労した。PASCAL VOCのDevKitのサンプルコードを読み解くくらい？

[クックパッドコンペ](https://deepanalytics.jp/contents/cookpad_dtc_tutorial)では画像毎の平均だったけど、一般的には(?)各クラスのAPの平均っぽい。

APも曲線をちゃんと積分するか0.1刻みに見るかがあって、PASCAL VOC 2007は後者っぽい。

### NMS(non-maximum suppression)

predict結果のうちconfidenceの高いものについて、それら同士で
同じクラスで重なっているものは一番confidenceの高いものを採用するという話。

論文上の説明がさらっとしてて最初は見落としていた。

### [重みを共有しているところがあると学習率をあまり大きく出来なさそう](https://twitter.com/ak11/status/916282847047983104)

学習の進みが遅かったりnanが出たりと困っていたけど、最終的にレイヤー毎に学習率を変えるようにした。

共有しているレイヤーは、学習率を `1/共有回数` 倍に。あとついでにベースネットワーク部分は `1/1000` 倍の学習率にした。
(最初はfreezeして追加部分だけ学習したほうが早いが、コードが増えたりfreeze解除時にメモリ不足で落ちると悲しいので。)
