#!/usr/bin/env python
"""事前学習をしてみるコード。"""
import argparse
import pathlib

import horovod.keras as hvd
import numpy as np
import sklearn.externals.joblib as joblib

import models
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
        tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log') if hvd.rank() == 0 else None)
        _run(args)


@tk.log.trace()
def _run(args):
    # モデルの読み込み
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')

    # データの読み込み
    y_train, y_test, _ = joblib.load(DATA_DIR / 'train_data.pkl')
    X_train = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_test)

    model = od.create_pretrain_network()
    gen = od.create_pretrain_generator()
    model = tk.dl.models.Model(model, gen, args.batch_size, use_horovod=True)
    model.compile(sgd_lr=0.5 / 256, loss='categorical_crossentropy', metrics=['acc'])

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    if hvd.rank() == 0:
        callbacks.append(tk.dl.callbacks.tsv_logger(RESULT_DIR / 'pretrain.history.tsv'))
        callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, np.zeros((len(X_train),)), validation_data=(X_test, np.zeros((len(X_test),))),
              epochs=args.epochs, callbacks=callbacks)
    model.save(RESULT_DIR / 'pretrain.model.h5')


if __name__ == '__main__':
    _main()
