#!/usr/bin/env python
"""前処理。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import data
import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', help='データの種類。', default='voc', choices=['voc', 'csv'])
    parser.add_argument('--base-network', help='ベースネットワークの種類。', default='resnet50', choices=['custom', 'vgg16', 'resnet50', 'xception'])
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320、512など。', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10, 5], type=int)
    args = parser.parse_args()
    tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log'))
    _run(args)


@tk.log.trace()
def _run(args):
    logger = tk.log.get(__name__)
    # データを読んでresults/配下にpklで保存し直す (学習時の高速化のため)
    logger.info('データの読み込み: data-dir=%s data-type=%s', DATA_DIR, args.data_type)
    (_, y_train), (X_test, y_test), class_names = data.load_data(DATA_DIR, args.data_type)
    joblib.dump(y_train, DATA_DIR / 'y_train.pkl')
    joblib.dump(y_test, DATA_DIR / 'y_test.pkl')
    joblib.dump(class_names, DATA_DIR / 'class_names.pkl')

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

    # customの場合、事前学習用データを作成
    if args.base_network == 'custom':
        with tk.dl.session():
            _make_teacher_data()


def _make_teacher_data():
    """事前学習用データの作成。"""
    import keras
    batch_size = 500
    data_path = DATA_DIR / 'cifar100_teacher_pred.pkl'
    if data_path.is_file():
        return
    (X_train, _), (X_test, _) = keras.datasets.cifar100.load_data()
    X = np.concatenate([X_train, X_test])

    teacher_model = keras.applications.Xception(include_top=False, input_shape=(71, 71, 3))
    teacher_model, batch_size = tk.dl.models.multi_gpu_model(teacher_model, batch_size)
    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.Resize((71, 71)))
    gen.add(tk.image.ProcessInput(keras.applications.xception.preprocess_input, batch_axis=True))
    steps = gen.steps_per_epoch(len(X), batch_size)

    pred_list = []
    with tk.tqdm(total=len(X), desc='make data') as pbar:
        for i, X_batch in enumerate(gen.flow(X, batch_size=batch_size)):
            pred_list.append(teacher_model.predict(X_batch))
            pbar.update(len(X_batch))
            if i + 1 >= steps:
                break

    pred_list = np.concatenate(pred_list)
    joblib.dump(pred_list, data_path)


if __name__ == '__main__':
    _main()
