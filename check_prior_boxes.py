#!/usr/bin/env python
"""PriorBoxのチェック。"""
import argparse
import pathlib
import time

import numpy as np

import config
import data_voc
import evaluation
import models
import pytoolkit as tk


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(config.BASE_DIR.joinpath('data')))  # sambaの問題のためのwork around...
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320 or 512', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10, 5], type=int)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.create_tee_logger(config.RESULT_DIR.joinpath(pathlib.Path(__file__).stem + '.log'), fmt=None)
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データの読み込み
    data_dir = pathlib.Path(args.data_dir)
    (X_train, y_train), (X_test, y_test) = data_voc.load_data(data_dir)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 訓練データからパラメータを適当に決める。
    od = models.ObjectDetector.create(args.input_size, args.map_sizes, len(data_voc.CLASS_NAMES), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))
    # prior boxのカバー度合いのチェック
    od.check_prior_boxes(logger, config.RESULT_DIR, y_test, data_voc.CLASS_NAMES)

    # ついでに、試しに回答を出力してみる
    check_true_size = 32
    evaluation.plot_truth(X_test[:check_true_size], y_test[:check_true_size], config.RESULT_DIR.joinpath('___ground_truth'))


if __name__ == '__main__':
    _main()
