# データの配置方法メモ

## PASCAL VOC

- [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- [VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

`./data/VOCdevkit/ ...` となるようにデータを配置する。

## CSV (独自形式)

- `./data/csv/train.csv`
- `./data/csv/test.csv`
- `./data/csv/class_names.txt`
- `./data/csv/images/[csvのfilename列の値]`

`class_names.txt` は、全てのクラス名を改行区切りで書いたテキストファイル。

`train.csv` と `test.csv` は、以下の列を持つ。

- filename
- left
- top
- right
- bottom
- class_name

1画像に2個以上のbounding boxがある場合はfilenameが重複した行が複数になる。

サンプル: `./data/csv.7z` ([Sansanデータセット](https://www.nii.ac.jp/dsc/idr/sansan/sansan.html))
