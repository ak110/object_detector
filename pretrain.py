#!/usr/bin/env python
"""事前学習をしてみるコード。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=100, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=64, type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log') if tk.dl.hvd.is_master() else None)
        _run(args)


@tk.log.trace()
def _run(args):
    # データの読み込み
    y_train, y_test, _ = joblib.load(DATA_DIR / 'train_data.pkl')
    X_train = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_test)

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model = od.create_model('pretrain', args.batch_size, weights='imagenet')

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    if tk.dl.hvd.is_master():
        callbacks.append(tk.dl.callbacks.tsv_logger(RESULT_DIR / 'pretrain.history.tsv'))
        callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, np.zeros((len(X_train),)), validation_data=(X_test, np.zeros((len(X_test),))),
              epochs=args.epochs, callbacks=callbacks)
    model.save(RESULT_DIR / 'pretrain.model.h5')


if __name__ == '__main__':
    _main()
