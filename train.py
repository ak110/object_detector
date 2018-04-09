#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import argparse
import pathlib

import horovod.keras as hvd
import sklearn.externals.joblib as joblib

import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=300, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    parser.add_argument('--no-lr-decay', help='learning rateを減衰させない。epochs // 2で停止する。', action='store_true', default=False)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log') if hvd.rank() == 0 else None)
        _run(args)


@tk.log.trace()
def _run(args):
    logger = tk.log.get(__name__)

    # データの読み込み
    y_train, y_test, _ = joblib.load(DATA_DIR / 'train_data.pkl')
    X_train = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_test)

    # モデルの読み込み
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model, lr_multipliers = od.create_network()
    model = tk.dl.models.Model(model, od.create_generator(), args.batch_size, use_horovod=True)

    # 学習済み重みの読み込み
    for warm_path in (RESULT_DIR / 'model.base.h5', RESULT_DIR / 'pretrain.model.h5'):
        if warm_path.is_file():
            model.load_weights(warm_path)
            logger.info('warm start: %s', warm_path.name)
            break

    model.compile(sgd_lr=0.5 / 256, lr_multipliers=lr_multipliers, loss=od.loss, metrics=od.metrics)

    callbacks = []
    if args.no_lr_decay:
        args.epochs //= 2
    else:
        callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    if hvd.rank() == 0:
        callbacks.append(tk.dl.callbacks.tsv_logger(RESULT_DIR / 'history.tsv'))
        callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=args.epochs, callbacks=callbacks)
    model.save(RESULT_DIR / 'model.h5')


if __name__ == '__main__':
    _main()
