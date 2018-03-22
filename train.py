#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import argparse
import pathlib

import horovod.keras as hvd
import numpy as np

import config
import data
import models
import pytoolkit as tk


def _main():
    hvd.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    parser.add_argument('--no-lr-decay', help='learning rateを減衰させない。epochs // 2で停止する。', action='store_true', default=False)
    args = parser.parse_args()

    logger = tk.log.get()
    if hvd.rank() == 0:
        logger.addHandler(tk.log.stream_handler())
        logger.addHandler(tk.log.file_handler(config.RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))

    _run(logger, args)


@tk.log.trace()
def _run(logger, args):
    # データの読み込み
    (X_train, y_train), (X_test, y_test), _ = data.load_data(config.DATA_DIR, 'pkl')
    logger.info('train, test = %d, %d', len(X_train), len(X_test))

    # モデルの読み込み
    od = models.ObjectDetector.load(config.RESULT_DIR / 'model.pkl')
    logger.info('mean objects / image = %f', od.mean_objets)
    logger.info('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.info('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.info('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.info('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))

    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        model, lr_multipliers = od.create_network()
        logger.info('network depth: %d', tk.dl.count_network_depth(model))
        logger.info('trainable params: %d', tk.dl.count_trainable_params(model))

        # 学習済み重みの読み込み
        base_model_path = config.RESULT_DIR / 'model.base.h5'
        pre_model_path = config.RESULT_DIR / 'pretrain.model.h5'
        if base_model_path.is_file():
            tk.dl.load_weights(model, base_model_path)
            logger.info('warm start: %s', base_model_path.name)
        elif pre_model_path.is_file():
            tk.dl.load_weights(model, pre_model_path)
            logger.info('warm start: %s', pre_model_path.name)

        # 学習率：
        # ・CIFARなどの分類ではlr 0.5、batch size 256くらいが多いのでその辺を基準に。
        # ・バッチサイズに比例させると良さそう？
        # ・損失関数がconf + loc + iouなのでちょっと調整した方が良さそう？
        base_lr = 0.5 * (args.batch_size / 256) * hvd.size() / 3

        with tk.log.trace_scope('model.compile'):
            opt = tk.dl.nsgd()(lr=base_lr, lr_multipliers=lr_multipliers)
            opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')
            model.compile(opt, od.loss, od.metrics)

        gen = od.create_generator()

        callbacks = []
        if args.no_lr_decay:
            args.epochs //= 2
        else:
            callbacks.append(tk.dl.learning_rate_callback())
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(tk.dl.tsv_log_callback(config.RESULT_DIR / 'history.tsv'))
            callbacks.append(tk.dl.logger_callback())
        callbacks.append(tk.dl.freeze_bn_callback(0.95))

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
