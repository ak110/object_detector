#!/usr/bin/env python
"""事前学習をしてみるコード。"""
import argparse
import pathlib

import horovod.keras as hvd
import sklearn.externals.joblib as joblib

import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _main():
    hvd.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=128, type=int)
    args = parser.parse_args()
    tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log') if hvd.rank() == 0 else None)
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run(args)


@tk.log.trace()
def _run(args):
    import keras
    logger = tk.log.get(__name__)

    # モデルの読み込み
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')

    # データの読み込み
    (X_train, _), (X_test, _) = keras.datasets.cifar100.load_data()
    y = joblib.load(DATA_DIR / 'cifar100_teacher_pred.pkl')
    y_train, y_test = y[:50000, ...], y[50000:, ...]
    logger.info('train, test = %d, %d', len(X_train), len(X_test))

    image_size = (96, 96)
    model = od.create_pretrain_network(image_size)

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(image_size))
    gen.add(tk.image.RandomColorAugmentors(probability=0.5))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(od.get_preprocess_input(), batch_axis=True))

    model = tk.dl.models.Model(model, gen, args.batch_size, use_horovod=True)
    model.compile(sgd_lr=0.5 / 256, loss='mse', metrics=['mae'])

    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate())
    callbacks.extend(model.horovod_callbacks())
    if hvd.rank() == 0:
        callbacks.append(tk.dl.callbacks.tsv_logger(RESULT_DIR / 'pretrain.history.tsv'))
        callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=args.epochs, callbacks=callbacks)
    model.save(RESULT_DIR / 'pretrain.model.h5')


if __name__ == '__main__':
    _main()
