#!/usr/bin/env python
"""前処理。"""
import argparse
import pathlib
import time

import numpy as np
import sklearn.externals.joblib

import config
import data
import evaluation
import models
import pytoolkit as tk


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(config.DATA_DIR))  # sambaの問題のためのwork around...
    parser.add_argument('--data-type', help='データの種類。', default='voc', choices=['voc', 'csv'])
    parser.add_argument('--base-network', help='ベースネットワークの種類。', default='vgg16', choices=['custom', 'vgg16', 'resnet50', 'xception'])
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320、512など。', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10, 5], type=int)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(config.RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データを読んでresults/配下にpklで保存し直す (学習時の高速化のため)
    logger.info('データの読み込み: data-dir=%s data-type=%s', args.data_dir, args.data_type)
    (_, y_train), (X_test, y_test), class_names = data.load_data(args.data_dir, args.data_type)
    sklearn.externals.joblib.dump(y_train, config.DATA_DIR / 'y_train.pkl')
    sklearn.externals.joblib.dump(y_test, config.DATA_DIR / 'y_test.pkl')
    sklearn.externals.joblib.dump(class_names, config.DATA_DIR / 'class_names.pkl')

    # 訓練データからパラメータを適当に決める
    logger.info('ハイパーパラメータ算出: base-network=%s input-size=%s map-sizes=%s classes=%d',
                 args.base_network, args.input_size, args.map_sizes, len(class_names))
    od = models.ObjectDetector.create(args.base_network, args.input_size, args.map_sizes, len(class_names), y_train)
    sklearn.externals.joblib.dump(od, str(config.RESULT_DIR / 'model.pkl'))

    # prior boxのカバー度合いのチェック
    logger.info('mean objects / image = %f', od.mean_objets)
    logger.info('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.info('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.info('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.info('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))
    od.check_prior_boxes(logger, config.RESULT_DIR, y_test, class_names)

    # ついでに、試しに回答を出力してみる
    check_true_size = 32
    evaluation.plot_truth(X_test[:check_true_size], y_test[:check_true_size],
                          class_names, config.RESULT_DIR / '___ground_truth')


if __name__ == '__main__':
    _main()
