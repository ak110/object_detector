#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import os
import pathlib
import time

import better_exceptions
import numpy as np
import sklearn.externals.joblib

import pytoolkit as tk
from evaluation import evaluate
from generator import Generator
from voc_data import load_data

_BATCH_SIZE = 16


def _main():
    import matplotlib as mpl
    mpl.use('Agg')
    better_exceptions.MAX_LENGTH = 128
    script_path = pathlib.Path(__file__).resolve()
    base_dir = script_path.parent
    os.chdir(str(base_dir))

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
    (X_train, _), (X_test, y_test) = load_data(data_dir, args.debug, _BATCH_SIZE)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        # モデルの読み込み
        od = sklearn.externals.joblib.load(str(result_dir.joinpath('model.pkl')))
        model, _ = od.create_network()
        model.load_weights(str(result_dir.joinpath('model.best.h5')), by_name=True)

        # 評価
        gen = Generator(image_size=od.image_size, od=od)
        evaluate(logger, od, model, gen, X_test, y_test, _BATCH_SIZE, -1, result_dir)


if __name__ == '__main__':
    _main()
