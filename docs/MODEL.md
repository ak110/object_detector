# モデルの詳細メモ

## ネットワーク

ベースネットワーク(VGG16/ResNet50など) + FPN風

参考にしたもの: [DSSD](https://arxiv.org/abs/1701.06659)、[DSOD](https://arxiv.org/abs/1708.01241)、[FPN](https://arxiv.org/abs/1612.03144)など。

## 活性化関数

[ELU](https://arxiv.org/abs/1511.07289) (ほぼ趣味)

## 初期化

最後以外は `he_uniform` 、最後は `zeros` 。 (ほぼ趣味)

最後をゼロにしておくと学習開始直後がちょっと安定する気がする。

## Prior box

使用するfeature mapは、40x40 ～ 5x5の4つ + 1x1など。

feature map毎に8種類のprior boxを出力する。

8種類のprior boxのサイズ・アスペクト比は、訓練データからKMeansを使用して決める。
IoUは重心が一致している想定で算出する。
([YOLOv2](https://arxiv.org/abs/1612.08242)のDimension Clusters。)

## 損失関数：分類

[Focal loss](https://arxiv.org/abs/1708.02002)

## 損失関数：bounding box

`L1-smooth loss`

`x, y, w, h` ではなく `x1, y1, x2, y2` でやっている。(怪)

## 損失関数：IoU

`binary crossentropy`

予測結果のboxと答えのboxのIoUを予測(回帰)して、分類のconfidenceと合わせて使用するもの。
([自分で考案したつもりがYOLOv1が既にやっていた](https://twitter.com/ak11/status/917901136782278656)。)

値の回帰でbinary crossentropyを使うのは理屈としては変だが、
経験上うまくいくのでやってしまっている。
(たぶん勾配が答えと予測値の差になるから)

合わせ方は加算（相加平均）、乗算（相乗平均）、調和平均とか試したけど乗算が僅差で良かったので採用。(怪)

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

predict結果のうちconfidenceの高いものについて、
それら同士で同じクラスで重なっているものは一番confidenceの高いものを採用するという話。

論文上の説明がさらっとしてて最初は見落としていた。

### 学習率

[重みを共有しているところがあると学習率をあまり大きく出来なさそう](https://twitter.com/ak11/status/916282847047983104)という話があり、
学習の進みが遅かったりnanが出たりと困っていたけど、
最終的にレイヤー毎に学習率を変えるようにした。

共有しているレイヤーは、学習率を `1/共有回数` 倍に。あとついでにベースネットワーク部分は `1/100` 倍の学習率にした。
(最初はfreezeして追加部分だけ学習したほうが早いが、コードが増えたりfreeze解除時にメモリ不足で落ちると悲しいので。)
