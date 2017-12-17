#!/usr/bin/env python
"""data/csvの前処理。

以下のファイルを読み込み、ObjectsAnnotationの配列に変換して保存する。

- ./data/csv/train.csv
- ./data/csv/test.csv
- ./data/csv/class_names.txt
- ./data/csv/images/[csvのfilename列の値]

`class_names.txt` は、csvに出現する全てのクラス名を改行区切りで書いたテキストファイル。

`train.csv` と `test.csv` は、filename,left,top,right,bottom,class_nameの列のみ使用する。

1画像に2個以上のbounding boxがある場合はfilenameが重複した行が複数になる。

"""
import pathlib

import numpy as np
import pandas as pd
import PIL.Image
import sklearn.externals.joblib
from tqdm import tqdm

import pytoolkit as tk


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    csv_dir = data_dir / 'csv'
    image_dir = csv_dir / 'images'

    # クラス名の一覧
    class_names = list(filter(lambda x: len(x) > 0, tk.io.read_all_lines(csv_dir / 'class_names.txt')))
    assert 'bg' not in class_names
    class_names = np.unique(['bg'] + class_names)
    class_name_to_id = {class_name: i for i, class_name in enumerate(class_names)}  # 'bg': 0
    sklearn.externals.joblib.dump(class_names, data_dir / 'class_names.pkl')

    for data_type in tqdm(('train', 'test'), desc='type', ncols=128, ascii=True):
        df = pd.read_csv(str(csv_dir / '{}.csv'.format(data_type)))
        # クラス名の一覧
        # ファイル名の一覧
        X = np.unique(df['filename'].values)
        y = []
        for x in tqdm(X, desc='images', ncols=128, ascii=True):
            # 画像サイズの取得
            with PIL.Image.open(image_dir / x) as img:
                width, height = img.size
            # class / bounding box
            df_target = df[df['filename'] == x]
            classes = [class_name_to_id[cn] for cn in df_target['class_name'].values]
            bboxes = [tpl for tpl in zip(df_target['left'].values / width, df_target['top'].values / height,
                                         df_target['right'].values / width, df_target['bottom'].values / height)]
            enables = [0 <= x1 < x2 < 1 and 0 <= y1 < y2 <= 1 for x1, y1, x2, y2 in bboxes]
            if sum(enables) <= 0:
                continue
            y.append(tk.ml.ObjectsAnnotation(
                folder='csv/images',  # data_dirからの相対パスにしておく
                filename=x,
                width=width,
                height=height,
                classes=[t for enable, t in zip(enables, classes) if enable],
                bboxes=[t for enable, t in zip(enables, bboxes) if enable],
                difficults=None))
        y = np.array(y)

        sklearn.externals.joblib.dump(y, data_dir / 'y_{}.pkl'.format(data_type))

    print('')
    print('finished.')

if __name__ == '__main__':
    _main()
