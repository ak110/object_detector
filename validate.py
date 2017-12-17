#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import pathlib
import time

import numpy as np
import sklearn.externals.joblib

import config
import data_voc
import evaluation
import generator
import pytoolkit as tk


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(base_dir.joinpath('data')))  # sambaの問題のためのwork around...
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.create_tee_logger(config.RESULT_DIR.joinpath(pathlib.Path(__file__).stem + '.log'), fmt=None)
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データの読み込み
    data_dir = pathlib.Path(args.data_dir)
    (X_train, _), (X_test, y_test) = data_voc.load_data(data_dir)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        # モデルの読み込み
        od = sklearn.externals.joblib.load(str(config.RESULT_DIR.joinpath('model.pkl')))
        model, _ = od.create_network()
        model.load_weights(str(config.RESULT_DIR.joinpath('model.h5')), by_name=True)

        # マルチGPU対応
        logger.debug('gpu count = %d', tk.get_gpu_count())
        model, batch_size = tk.dl.create_data_parallel_model(model, args.batch_size)

        # 評価
        gen = generator.Generator(image_size=od.image_size, od=od)
        evaluation.evaluate(logger, od, model, gen, X_test, y_test, batch_size, -1, config.RESULT_DIR)


if __name__ == '__main__':
    _main()
