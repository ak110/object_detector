#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import argparse
import pathlib

import horovod.keras as hvd

import data
import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'


def _main():
    tk.better_exceptions()
    hvd.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    parser.add_argument('--no-lr-decay', help='learning rateを減衰させない。epochs // 2で停止する。', action='store_true', default=False)
    args = parser.parse_args()
    tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log') if hvd.rank() == 0 else None)
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run(args)


@tk.log.trace()
def _run(args):
    logger = tk.log.get(__name__)
    # データの読み込み
    (X_train, y_train), (X_test, y_test), _ = data.load_data(DATA_DIR, 'pkl')

    # モデルの読み込み
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model, lr_multipliers = od.create_network()

    # 学習済み重みの読み込み
    for warm_path in (RESULT_DIR / 'model.base.h5', RESULT_DIR / 'pretrain.model.h5'):
        if warm_path.is_file():
            tk.dl.load_weights(model, warm_path)
            logger.info('warm start: %s', warm_path.name)
            break

    # 学習率：
    # ・CIFARなどの分類ではlr 0.5、batch size 256くらいが多いのでその辺を基準に。
    # ・バッチサイズに比例させると良さそう？
    # ・損失関数がconf + loc + iouなのでちょっと調整した方が良さそう？
    base_lr = 0.5 * (args.batch_size / 256) * hvd.size() / 3

    with tk.log.trace_scope('model.compile'):
        opt = tk.dl.nsgd()(lr=base_lr, lr_multipliers=lr_multipliers)
        opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')
        model.compile(opt, od.loss, od.metrics)

    callbacks = []
    if args.no_lr_decay:
        args.epochs //= 2
    else:
        callbacks.append(tk.dl.learning_rate_callback())
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(RESULT_DIR / 'history.tsv'))
        callbacks.append(tk.dl.logger_callback())
    callbacks.append(tk.dl.freeze_bn_callback(0.95))

    gen = od.create_generator()
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=args.batch_size, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(len(X_train), args.batch_size) // hvd.size(),
        epochs=args.epochs,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=gen.flow(X_test, y_test, batch_size=args.batch_size, shuffle=True),
        validation_steps=gen.steps_per_epoch(len(X_test), args.batch_size) // hvd.size(),  # horovodのサンプルでは * 3 だけど早く進んで欲しいので省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(RESULT_DIR / 'model.h5'))


if __name__ == '__main__':
    _main()
