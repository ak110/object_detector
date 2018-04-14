# モデルの詳細メモ

## ネットワーク

ベースネットワーク(VGG16/ResNet50など) + FPN風

参考にしたもの: [DSSD](https://arxiv.org/abs/1701.06659)、[DSOD](https://arxiv.org/abs/1708.01241)、[FPN](https://arxiv.org/abs/1612.03144)、[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)など。

## 活性化関数

[ELU](https://arxiv.org/abs/1511.07289) (ほぼ趣味)

## 初期化

最後以外は `he_uniform` 、最後は `zeros` 。 (ほぼ趣味)

最後をゼロにしておくと学習開始直後がちょっと安定する気がする。

## Prior box

使用するfeature mapは、40x40 ～ 10x10の3つなど。

feature map毎に8種類のprior boxを出力する。

8種類のprior boxのサイズ・アスペクト比は、訓練データからKMeansを使用して決める。
IoUは重心が一致している想定で算出する。
([YOLOv2](https://arxiv.org/abs/1612.08242)のDimension Clusters。)

## 損失関数：Objectness score

YOLOv3風に物体か否かを2クラス分類。

cross entropyではなく[Focal loss](https://arxiv.org/abs/1708.02002)を使用。

## 損失関数：classification

`binary crossentropy`

YOLOv3風に。(普通の分類では大差無いと思う)

## 損失関数：bounding box

`L1-smooth loss`

`x, y, w, h` ではなく `x1, y1, x2, y2` でやっている。(怪)

## confidence

NMSするときとかに使うconfidenceは、Objectness scoreとclassificationの幾何平均にしている。

(調和平均が正しいと思うのだけど調和平均より算術平均より幾何平均が良さそう。)

## DataAugmentation

[Random Erasing](https://arxiv.org/abs/1708.04896)ほか手当たり次第に。

Random Erasingは、あまりbounding boxを隠しすぎないように色々手動で条件を付けてみている。

## 作ってて困ったポイント備忘録

困ったら[SSDのKeras実装](https://github.com/rykov8/ssd_keras)と[ChainerCV](https://github.com/chainer/chainercv)を見たら大体解決した。ありがたい。

### mAP

計算方法の情報が見つからなくて苦労した。PASCAL VOCのDevKitのサンプルコードを読み解くくらい？

[クックパッドコンペ](https://deepanalytics.jp/contents/cookpad_dtc_tutorial)では画像毎の平均だったけど、一般的には(?)各クラスのAPの平均っぽい。

APも曲線をちゃんと積分するか0.1刻みに見るかがあって、PASCAL VOC 2007は後者っぽい。

### NMS(non-maximum suppression)

predict結果のうちconfidenceの高いものについて、それら同士で同じクラスで重なっているものは一番confidenceの高いものを採用するという話。

論文上の説明がさらっとしてて最初は見落としていた。

### 学習率

[重みを共有しているところがあると学習率をあまり大きく出来なさそう](https://twitter.com/ak11/status/916282847047983104)という話があり、
学習の進みが遅かったりnanが出たりと困っていたけど、最終的にレイヤー毎に学習率を変えるようにした。

共有しているレイヤーは、学習率を `1/共有回数` 倍に。あとついでにベースネットワーク部分は `1/100` 倍の学習率にした。
(最初はfreezeして追加部分だけ学習したほうが早いが、コードが増えたりfreeze解除時にメモリ不足で落ちると悲しいので。)
