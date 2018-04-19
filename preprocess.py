#!/usr/bin/env python
"""前処理。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', help='データの種類。', default='voc', choices=['voc', 'csv'])
    parser.add_argument('--base-network', help='ベースネットワークの種類。', default='vgg16', choices=['custom', 'vgg16', 'resnet50', 'xception'])
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320、512など。', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10], type=int)
    args = parser.parse_args()
    tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log'))
    _run(args)


@tk.log.trace()
def _run(args):
    logger = tk.log.get(__name__)
    # データを読んでresults/配下にpklで保存し直す (学習時の高速化のため)
    logger.info('データの読み込み: data-dir=%s data-type=%s', DATA_DIR, args.data_type)
    (_, y_train), (X_test, y_test), class_names = load_data(DATA_DIR, args.data_type)
    joblib.dump((y_train, y_test, class_names), DATA_DIR / 'train_data.pkl')

    # 訓練データからパラメータを適当に決める
    od = models.ObjectDetector.create(args.base_network, args.input_size, args.map_sizes, len(class_names), y_train)
    od.save(RESULT_DIR / 'model.pkl')
    od.summary()
    od.check_prior_boxes(y_test, class_names)

    # ついでに、試しに回答を出力してみる
    check_true_size = 32
    for X, y in zip(X_test[:check_true_size], y_test[:check_true_size]):
        tk.ml.plot_objects(
            X, RESULT_DIR / '___ground_truth' / (pathlib.Path(X).name + '.png'),
            y.classes, None, y.bboxes, class_names)


@tk.log.trace()
def load_data(data_dir, data_type: str):
    """データの読み込み"""
    assert data_type in ('voc', 'csv')
    data_dir = pathlib.Path(data_dir)

    if data_type == 'voc':
        y_train, y_test, class_names = load_data_voc(data_dir)
    else:
        y_train, y_test, class_names = load_data_csv(data_dir)

    X_train = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_test)

    logger = tk.log.get(__name__)
    logger.info('train, test = %d, %d', len(X_train), len(X_test))

    return (X_train, y_train), (X_test, y_test), class_names


def load_data_voc(data_dir: pathlib.Path):
    """データの読み込み"""
    class_name_to_id = {n: i for i, n in enumerate(tk.ml.VOC_CLASS_NAMES)}
    y_train = tk.ml.ObjectsAnnotation.load_voc_0712_trainval(data_dir, class_name_to_id, without_difficult=True)
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(data_dir, class_name_to_id)
    return y_train, y_test, tk.ml.VOC_CLASS_NAMES


def load_data_csv(data_dir: pathlib.Path):
    """データの読み込み"""
    csv_dir = data_dir / 'csv'
    image_dir = csv_dir / 'images'
    class_names_path = csv_dir / 'class_names.txt'

    # クラス名の一覧
    class_names = list(filter(lambda x: len(x) > 0, tk.io.read_all_lines(class_names_path)))
    class_names = np.unique(class_names).tolist()
    class_name_to_id = {class_name: i for i, class_name in enumerate(class_names)}

    y_train = _load_csv_annotations(csv_dir, 'train', image_dir, class_name_to_id)
    y_test = _load_csv_annotations(csv_dir, 'test', image_dir, class_name_to_id)

    return y_train, y_test, class_names


def _load_csv_annotations(csv_dir, data_type, image_dir, class_name_to_id):
    import pandas as pd
    df = pd.read_csv(str(csv_dir / f'{data_type}.csv'))
    # クラス名の一覧
    # ファイル名の一覧
    X = np.unique(df['filename'].values)
    y = []
    for x in tk.tqdm(X, desc='images'):
        # 画像サイズの取得
        import PIL.Image
        with PIL.Image.open(image_dir / x) as img:
            width, height = img.size
        # class / bounding box
        df_target = df[df['filename'] == x]
        classes = [class_name_to_id[cn] for cn in df_target['class_name'].values]
        bboxes = [tpl for tpl in zip(df_target['left'].values / width, df_target['top'].values / height,
                                     df_target['right'].values / width, df_target['bottom'].values / height)]
        enables = [0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 for x1, y1, x2, y2 in bboxes]
        if not all(enables):
            import warnings
            warnings.warn(f'不正なデータ:\n{df_target}')
        if not any(enables):
            continue
        y.append(tk.ml.ObjectsAnnotation(
            folder='csv/images',  # data_dirからの相対パスにしておく
            filename=x,
            width=width,
            height=height,
            classes=[t for enable, t in zip(enables, classes) if enable],
            bboxes=[t for enable, t in zip(enables, bboxes) if enable],
            difficults=None))
    return np.array(y)


if __name__ == '__main__':
    _main()
