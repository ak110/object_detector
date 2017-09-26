#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
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
from model import ObjectDetector
from net import create_network
from voc_data import CLASS_NAMES, load_data

_BATCH_SIZE = 16
_MAX_EPOCH = 300
_MAX_EPOCH_DEBUG = 16
_BASE_LR = 1e-1  # 1e-3


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
    parser.add_argument('--pretrained', help='pretrained start。', action='store_true', default=False)
    parser.add_argument('--warm', help='warm start。', action='store_true', default=False)
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

    # 訓練データからパラメータを適当に決める。
    od = ObjectDetector.create(len(CLASS_NAMES), y_train)
    sklearn.externals.joblib.dump(od, str(result_dir.joinpath('model.pkl')))
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d', len(od.pb_locs))

    import keras
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        model = create_network(od)
        model.summary(print_fn=logger.debug)
        logger.debug('network depth: %d', tk.dl.count_network_depth(model))
        tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

        # keras.utils.plot_model(model, str(result_dir.joinpath('model.png')))

        # 学習済み重みの読み込み
        if args.pretrained:
            model_path = result_dir.parent.joinpath('results_pretrain', 'model.h5')
            tk.dl.load_weights(model, model_path)
            logger.debug('load pretrained weights: %s', model_path.name)
        elif args.warm:
            model_path = result_dir.joinpath('model.best.h5')
            tk.dl.load_weights(model, model_path)
            logger.debug('warm start: %s', model_path.name)

        # マルチGPU対応
        gpu_count = tk.get_gpu_count()
        logger.debug('gpu count = %d', gpu_count)
        if gpu_count <= 1:
            batch_size = _BATCH_SIZE
        else:
            model = tk.dl.create_data_parallel_model(model, gpu_count)
            batch_size = _BATCH_SIZE * gpu_count
            keras.utils.plot_model(model, str(result_dir.joinpath('model.multigpu.png')), show_shapes=True)

        # model.compile('nadam', od.loss, [od.loss_loc, od.acc_bg, od.acc_obj])
        model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), od.loss, [od.loss_loc, od.acc_bg, od.acc_obj])

        gen = Generator(image_size=od.input_size, od=od)

        max_epoch = _MAX_EPOCH_DEBUG if args.debug else _MAX_EPOCH
        lr_list = [_BASE_LR] * (max_epoch // 2) + [_BASE_LR / 10] * (max_epoch // 4) + [_BASE_LR / 100] * (max_epoch // 4)

        callbacks = []
        callbacks.append(tk.dl.my_callback_factory()(result_dir, lr_list=lr_list))
        # callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=_BASE_LR, beta1=0.9990, beta2=0.9995, margin_iterations=518))
        callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
        callbacks.append(keras.callbacks.ModelCheckpoint(str(result_dir.joinpath('model.best.h5')), save_best_only=True))
        # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
        # if K.backend() == 'tensorflow':
        #     callbacks.append(keras.callbacks.TensorBoard())

        # 各epoch毎にmAPを算出して表示してみる
        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: evaluate(logger, od, model, gen, X_test, y_test, _BATCH_SIZE, epoch, result_dir)
        ))

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=batch_size, data_augmentation=not args.debug, shuffle=True),
            steps_per_epoch=gen.steps_per_epoch(len(X_train) * (512 if args.debug else 1), batch_size),
            epochs=max_epoch,
            validation_data=gen.flow(X_test, y_test, batch_size=batch_size),
            validation_steps=gen.steps_per_epoch(len(X_test), batch_size),
            callbacks=callbacks)

        model.save(str(result_dir.joinpath('model.h5')))

        # 最終結果表示
        evaluate(logger, od, model, gen, X_test, y_test, _BATCH_SIZE, None, result_dir)


if __name__ == '__main__':
    _main()
