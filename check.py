#!/usr/bin/env python
"""PriorBoxのチェック。"""
import argparse
import os
import pathlib
import time

import better_exceptions
import numpy as np

import pytoolkit as tk
from evaluation import plot_truth
from model import ObjectDetector
from voc_data import CLASS_NAMES, load_data

_BATCH_SIZE = 16


def _main():
    import matplotlib as mpl
    mpl.use('Agg')
    better_exceptions.MAX_LENGTH = 128
    script_path = pathlib.Path(__file__).resolve()
    base_dir = script_path.parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='デバッグモード。', action='store_true', default=False)
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(base_dir.joinpath('data')))  # sambaの問題のためのwork around...
    args = parser.parse_args()

    result_dir = base_dir.joinpath('results{}'.format('_debug' if args.debug else ''))
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data_dir)
    logger = tk.create_tee_logger(result_dir.joinpath(script_path.stem + '.log'))

    start_time = time.time()
    _run(args, logger, result_dir, data_dir)
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(args, logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    # データの読み込み
    (X_train, y_train), (X_test, y_test) = load_data(data_dir, args.debug, _BATCH_SIZE)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 試しに回答を出力してみる。
    plot_truth(X_test[:_BATCH_SIZE], y_test[:_BATCH_SIZE], result_dir.joinpath('___ground_truth'))
    # for i, y in enumerate(y_test[:3]):
    #     logger.debug('y_test[%d]: %s', i, str(y))

    # 訓練データからパラメータを適当に決める。
    od = ObjectDetector.create(len(CLASS_NAMES), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d', len(od.pb_locs))

    # prior boxのカバー度合いのチェック
    od.check_prior_boxes(logger, result_dir, y_test, CLASS_NAMES)


if __name__ == '__main__':
    _main()
