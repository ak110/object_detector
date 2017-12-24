#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import argparse
import pathlib
import time

import horovod.keras as hvd
import numpy as np
import sklearn.externals.joblib

import config
import data
import generator
import pytoolkit as tk


def _main():
    hvd.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=12, type=int)
    parser.add_argument('--no-lr-decay', help='learning rateを減衰させない。epochs // 2で停止する。', action='store_true', default=False)
    args = parser.parse_args()

    start_time = time.time()
    logger = tk.log.get()
    if hvd.rank() == 0:
        logger.addHandler(tk.log.stream_handler())
        logger.addHandler(tk.log.file_handler(config.RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))
    _run(logger, args)
    elapsed_time = time.time() - start_time
    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, args):
    # データの読み込み
    (X_train, y_train), (X_test, y_test), _ = data.load_data(config.DATA_DIR, 'pkl')
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # モデルの読み込み
    od = sklearn.externals.joblib.load(str(config.RESULT_DIR / 'model.pkl'))  # type: models.ObjectDetector
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.debug('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.debug('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))

    import keras
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        model, lr_multipliers = od.create_network()
        if hvd.rank() == 0:
            with (config.RESULT_DIR / 'network.txt').open('w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.debug('network depth: %d', tk.dl.count_network_depth(model))

        # 学習済み重みの読み込み
        base_model_path = config.RESULT_DIR / 'model.base.h5'
        if base_model_path.is_file():
            tk.dl.load_weights(model, base_model_path)
            logger.debug('warm start: %s', base_model_path.name)

        # 学習率：
        # ・CIFARなどの分類ではlr 0.5、batch size 256くらいが多いのでその辺を基準に。
        # ・バッチサイズに比例させると良さそう？
        # ・lossがクラス、位置×4、IoUなので / 3くらいしてみる。(適当)
        base_lr = 0.5 / 3 * (args.batch_size / 256) * hvd.size()

        opt = tk.dl.nsgd()(lr=base_lr, lr_multipliers=lr_multipliers)
        opt = hvd.DistributedOptimizer(opt)
        model.compile(opt, od.loss, od.metrics)

        gen = generator.Generator(od.image_size, od.get_preprocess_input(), od)

        callbacks = []
        if args.no_lr_decay:
            args.epochs //= 2
        else:
            callbacks.append(tk.dl.learning_rate_callback(lr=base_lr, epochs=args.epochs))
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(keras.callbacks.ModelCheckpoint(str(config.RESULT_DIR / 'model.h5'), period=16, verbose=1))
            callbacks.append(tk.dl.tsv_log_callback(config.RESULT_DIR / 'history.tsv'))
            callbacks.append(tk.dl.logger_callback())

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=args.batch_size, data_augmentation=True, shuffle=True),
            steps_per_epoch=gen.steps_per_epoch(len(X_train), args.batch_size) // hvd.size(),
            epochs=args.epochs,
            verbose=1 if hvd.rank() == 0 else 0,
            validation_data=gen.flow(X_test, y_test, batch_size=args.batch_size, shuffle=True),
            validation_steps=gen.steps_per_epoch(len(X_test), args.batch_size) // hvd.size(),  # horovodのサンプルでは * 3 だけど早く進んで欲しいので省略
            callbacks=callbacks)

        if hvd.rank() == 0:
            model.save(str(config.RESULT_DIR / 'model.h5'))


if __name__ == '__main__':
    _main()
