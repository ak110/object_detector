#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import argparse
import pathlib

import sklearn.externals.joblib as joblib

import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=300, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
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

    # 読み込む重みの決定
    weights = 'imagenet'
    for warm_path in (RESULT_DIR / 'model.base.h5', RESULT_DIR / 'pretrain.model.h5'):
        if warm_path.is_file():
            weights = weights
            break

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model = od.create_model('train', args.batch_size, weights)

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate(reduce_epoch_rates=(0.5, 0.75, 0.875)))
    callbacks.extend(model.horovod_callbacks())
    if tk.dl.hvd.is_master():
        callbacks.append(tk.dl.callbacks.tsv_logger(RESULT_DIR / 'history.tsv'))
        callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=args.epochs, callbacks=callbacks)
    model.save(RESULT_DIR / 'model.h5')


if __name__ == '__main__':
    _main()
