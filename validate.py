#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import pathlib
import time

import numpy as np
import sklearn.externals.joblib

import config
import data
import evaluation
import generator
import pytoolkit as tk


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(config.DATA_DIR))  # sambaの問題のためのwork around...
    parser.add_argument('--data-type', help='データの種類。', default='pkl', choices=['voc', 'csv', 'pkl'])
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(config.RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データの読み込み
    (X_train, _), (X_test, y_test), class_names = data.load_data(args.data_dir, args.data_type)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    with tk.dl.session():
        # モデルの読み込み
        od = sklearn.externals.joblib.load(str(config.RESULT_DIR / 'model.pkl'))  # type: models.ObjectDetector
        model, _ = od.create_network()
        model.load_weights(str(config.RESULT_DIR / 'model.h5'))
        # マルチGPU対応
        logger.debug('gpu count = %d', tk.get_gpu_count())
        model, batch_size = tk.dl.create_data_parallel_model(model, args.batch_size)

        # 評価
        gen = generator.Generator(od.image_size, od.get_preprocess_input(), od)
        evaluation.evaluate(logger, od, model, gen, X_test, y_test, batch_size, -1, class_names, config.RESULT_DIR)


if __name__ == '__main__':
    _main()
