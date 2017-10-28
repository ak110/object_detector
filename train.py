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
from voc_data import CLASS_NAMES, load_data


def _main():
    import matplotlib as mpl
    mpl.use('Agg')
    better_exceptions.MAX_LENGTH = 128
    script_path = pathlib.Path(__file__).resolve()
    base_dir = script_path.parent
    os.chdir(str(base_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='デバッグモード。', action='store_true', default=False)
    parser.add_argument('--warm', help='warm start。', action='store_true', default=False)
    parser.add_argument('--data-dir', help='データディレクトリ。', default=str(base_dir.joinpath('data')))  # sambaの問題のためのwork around...
    parser.add_argument('--input-size', help='入力画像の一辺のサイズ。320 or 512', default=320, type=int)
    parser.add_argument('--map-sizes', help='prior boxの一辺の数。', nargs='+', default=[40, 20, 10, 5], type=int)
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=12, type=int)
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
    (X_train, y_train), (X_test, y_test) = load_data(data_dir, args.debug, args.batch_size)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 訓練データからパラメータを適当に決める。
    od = ObjectDetector.create(args.input_size, args.map_sizes, len(CLASS_NAMES), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))

    sklearn.externals.joblib.dump(od, str(result_dir.joinpath('model.pkl')))

    import keras
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        model, lr_multipliers = od.create_network()
        model.summary(print_fn=logger.debug)
        logger.debug('network depth: %d', tk.dl.count_network_depth(model))
        tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))

        # keras.utils.plot_model(model, str(result_dir.joinpath('model.png')))

        # 学習済み重みの読み込み
        if args.warm:
            model_path = result_dir.joinpath('model.h5')
            tk.dl.load_weights(model, model_path)
            logger.debug('warm start: %s', model_path.name)

        # マルチGPU対応
        logger.debug('gpu count = %d', tk.get_gpu_count())
        model, batch_size = tk.dl.create_data_parallel_model(model, args.batch_size)

        model.compile(tk.dl.nsgd()(lr_multipliers=lr_multipliers), od.loss, od.metrics)

        gen = Generator(image_size=od.image_size, od=od)

        # 学習率の決定：
        # ・CIFARやImageNetの分類では
        #   lr 0.1～0.5、batch size 64～256くらいが多いのでその辺を基準に。
        # ・バッチサイズのsqrtに比例させると良さそう
        #   (cf. https://www.slideshare.net/JiroNishitoba/20170629)
        # ・lossが分類＋回帰×4＋回帰なので、とりあえず1 / 3倍にしてみる。(怪)
        # epoch数：
        # ・指定値 // 2 + 指定値 // 4 + 指定値 // 4。
        base_lr = 1e-1 / 3 * np.sqrt(batch_size) / np.sqrt(128)
        lr_list = [base_lr] * (args.epochs // 2) + [base_lr / 10] * (args.epochs // 4) + [base_lr / 100] * (args.epochs // 4)
        epochs = len(lr_list)

        callbacks = []
        callbacks.append(tk.dl.my_callback_factory()(result_dir, lr_list=lr_list))
        callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: _on_epoch_end(
                logger, od, model, gen, X_test, y_test, batch_size, epoch, epochs, result_dir)
        ))

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=batch_size, data_augmentation=not args.debug, shuffle=True),
            steps_per_epoch=gen.steps_per_epoch(len(X_train) * (512 if args.debug else 1), batch_size),
            epochs=epochs,
            validation_data=gen.flow(X_test, y_test, batch_size=batch_size),
            validation_steps=gen.steps_per_epoch(len(X_test), batch_size),
            callbacks=callbacks)

        # 最終結果表示
        evaluate(logger, od, model, gen, X_test, y_test, batch_size, epochs - 1, result_dir)


def _on_epoch_end(logger, od, model, gen, X_test, y_test, batch_size, epoch, epochs, result_dir):
    """モデルをsaveして、`mAP`を算出してprintする。"""
    if not ((epoch + 1) % 16 == 0 or (epoch + 1) & epoch == 0):
        return  # 重いので16回あたり1回だけ実施
    if epoch + 1 == epochs:
        return  # 最後の場合はfit_generatorが終わってから改めてやる

    model.save(str(result_dir.joinpath('model.h5')))

    print('', file=sys.stdout, flush=True)
    evaluate(logger, od, model, gen, X_test, y_test, batch_size, epoch, result_dir)
    print('', file=sys.stderr, flush=True)


if __name__ == '__main__':
    _main()
