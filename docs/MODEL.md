# モデルの詳細メモ

基本的にはFPSよりも精度重視な感じ。

## ネットワーク

小さくしてからまた大きくするFPN風なもの。
prediction moduleの重み共有もあり。

参考にしたもの:
[RetinaNet](https://arxiv.org/abs/1708.02002)、
[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
など。

## ベースネットワーク

Darknet53(YOLOv3のベースネットワーク)が[強かった](https://twitter.com/ak11/status/1002472881387737088)。

## 活性化関数

[ELU](https://arxiv.org/abs/1511.07289) (ほぼ趣味)

## Prior box

使用するfeature mapは、40x40, 20x20, 10x10の3つ。

feature map毎に8種類のprior boxを出力する。

8種類のprior boxのサイズ・アスペクト比は、訓練データからKMeansを使用して決める。
[YOLOv2](https://arxiv.org/abs/1612.08242)のDimension Clustersのようなものだが、
距離は `1 - IoU` ではなく、feature mapの格子のサイズに対する相対値をそのままユークリッド距離でKMeansしている。

## 損失関数：Objectness score

YOLOv3風に物体か否かを2クラス分類。

cross entropyではなく[Focal loss](https://arxiv.org/abs/1708.02002)を使用。

## 損失関数：classification

普通にsoftmax + categorical crossentropy。

YOLOv3のsigmoid + binary crossentropyもやってみたが、
クラス数の増減で正負のバランスが変わってしまうのが気になるのでやめた。

## 損失関数：bounding box

mean squared error。

普通にL1-smooth lossでもよいが、大差なかったので。

`x, y, w, h` ではなく `x1, y1, x2, y2` でやっている。(怪)

## confidence

NMSするときに使うconfidenceは、Objectness scoreとclassificationの掛け算にしている。
やってみるとなぜか算術平均や調和平均よりmAPが高くなるため。(怪)
幾何平均だと1.0寄りになってしまうので√せず掛け算だけにした。

## DataAugmentation

[Random Erasing](https://arxiv.org/abs/1708.04896)ほか手当たり次第に。

Random Erasingは、あまりbounding boxを隠しすぎないように色々手動で条件を付けてみている。

## 作ってて困ったポイント備忘録

困ったら[SSDのKeras実装](https://github.com/rykov8/ssd_keras)と[ChainerCV](https://github.com/chainer/chainercv)を見たら大体解決した。ありがたい。

### mAP

計算方法の情報が見つからなくて苦労した。PASCAL VOCのDevKitのサンプルコードを読み解くくらい？

[クックパッドコンペ](https://signate.jp/competitions/31#evaluation)では画像毎の平均だったけど、一般的には(?)各クラスのAPの平均っぽい。

APも曲線をちゃんと積分するか0.1刻みに見るかがあって、PASCAL VOC 2007は後者っぽい。

### NMS(non-maximum suppression)

predict結果のうちconfidenceの高いものについて、それら同士で同じクラスで重なっているものは一番confidenceの高いものを採用するという話。

論文上の説明がさらっとしてて最初は見落としていた。

### 学習率

[重みを共有しているところがあると学習率をあまり大きく出来なさそう](https://twitter.com/ak11/status/916282847047983104)という話があり、
学習の進みが遅かったりnanが出たりと困っていたけど、最終的にレイヤー毎に学習率を変えるようにした。

共有しているレイヤーは、学習率を `1/共有回数` 倍に。あとついでにベースネットワーク部分は `1/100` 倍の学習率にした。
(最初はfreezeして追加部分だけ学習したほうが早いが、コードが増えたりfreeze解除時にメモリ不足で落ちると悲しいので。)
