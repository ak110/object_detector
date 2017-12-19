#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
import argparse
import pathlib
import time

import numpy as np
import sklearn.externals.joblib

import config
import data
import evaluation
import generator
import models
import pytoolkit as tk


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm', help='warm start。', action='store_true', default=False)
    parser.add_argument('--data', help='データの種類。', default='voc', choices=['voc', 'pkl'])
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(config.BASE_DIR.joinpath('data')))  # sambaの問題のためのwork around...
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320、512など。', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10, 5], type=int)
    parser.add_argument('--network', help='ベースネットワークの種類。', default='resnet50', choices=['custom', 'vgg16', 'resnet50', 'xception'])
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=12, type=int)
    parser.add_argument('--no-lr-decay', help='learning rateを減衰させない。epochs // 2で停止する。', action='store_true', default=False)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.create_tee_logger(config.RESULT_DIR.joinpath(pathlib.Path(__file__).stem + '.log'), fmt=None)
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データの読み込み
    data_dir = pathlib.Path(args.data_dir)
    (X_train, y_train), (X_test, y_test), class_names = data.load_data(data_dir, args.data)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 訓練データからパラメータを適当に決める。
    od = models.ObjectDetector.create(args.input_size, args.map_sizes, len(class_names), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))
    sklearn.externals.joblib.dump(od, str(config.RESULT_DIR.joinpath('model.pkl')))

    import keras
    with tk.dl.session():
        gpu_count = tk.get_gpu_count()
        with tk.dl.device(cpu=gpu_count >= 2):
            model, lr_multipliers = od.create_network(args.network)
        model.summary(print_fn=logger.debug)
        logger.debug('network depth: %d', tk.dl.count_network_depth(model))
        tk.dl.plot_model_params(model, config.RESULT_DIR.joinpath('model.params.png'))

        # keras.utils.plot_model(model, str(config.RESULT_DIR.joinpath('model.png')))

        # 学習済み重みの読み込み
        if args.warm:
            model_path = config.RESULT_DIR.joinpath('model.h5')
            tk.dl.load_weights(model, model_path)
            logger.debug('warm start: %s', model_path.name)

        # マルチGPU対応
        model, batch_size = tk.dl.create_data_parallel_model(model, args.batch_size)
        logger.debug('gpu count = %d', gpu_count)

        model.compile(tk.dl.nsgd()(lr_multipliers=lr_multipliers), od.loss, od.metrics)

        gen = generator.Generator(image_size=od.image_size, od=od, base_network=args.network)

        # 学習率：
        # ・CIFARやImageNetの分類ではlr 0.5、batch size 256くらいが多いのでその辺を基準に。
        # ・バッチサイズに比例させると良さそう？
        # ・lossがクラス、位置×4、IoUなので / 6くらいしてみる。
        # epoch数：
        # ・指定値 // 2 + 指定値 // 4 + 指定値 // 4。
        base_lr = 0.5 / 6 * (batch_size / 256)
        if args.no_lr_decay:
            lr_list = [base_lr] * (args.epochs // 2)
        else:
            lr_list = [base_lr] * (args.epochs // 2) + [base_lr / 10] * (args.epochs // 4) + [base_lr / 100] * (args.epochs // 4)
        epochs = len(lr_list)

        callbacks = []
        callbacks.append(tk.dl.my_callback_factory()(config.RESULT_DIR, lr_list=lr_list))
        callbacks.append(tk.dl.learning_curve_plotter_factory()(config.RESULT_DIR.joinpath('history.{metric}.png'), 'loss'))
        callbacks.append(keras.callbacks.ModelCheckpoint(str(config.RESULT_DIR.joinpath('model.h5')), save_weights_only=True, period=16, verbose=1))

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=batch_size, data_augmentation=True, shuffle=True),
            steps_per_epoch=gen.steps_per_epoch(len(X_train), batch_size),
            epochs=epochs,
            validation_data=gen.flow(X_test, y_test, batch_size=batch_size),
            validation_steps=gen.steps_per_epoch(len(X_test), batch_size),
            callbacks=callbacks)
        model.save_weights(str(config.RESULT_DIR.joinpath('model.h5')))

        evaluation.evaluate(logger, od, model, gen, X_test, y_test, batch_size, epochs, class_names, config.RESULT_DIR)

if __name__ == '__main__':
    _main()
