#!/usr/bin/env python
"""事前学習をしてみるコード。"""
import argparse
import pathlib

import better_exceptions
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
    better_exceptions.MAX_LENGTH = 100
    hvd.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=128, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=128, type=int)
    args = parser.parse_args()

    logger = tk.log.get()
    if hvd.rank() == 0:
        logger.addHandler(tk.log.stream_handler())
        logger.addHandler(tk.log.file_handler(RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))

    _run(logger, args)


@tk.log.trace()
def _run(logger, args):
    # モデルの読み込み
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    logger.info('mean objects / image = %f', od.mean_objets)
    logger.info('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.info('prior box aspect ratios = %s', str(od.pb_aspect_ratios))
    logger.info('prior box sizes = %s', str(np.unique([c['size'] for c in od.pb_info])))
    logger.info('prior box count = %d (valid=%d)', len(od.pb_mask), np.count_nonzero(od.pb_mask))

    import keras
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        # データの読み込み
        (X_train, _), (X_test, _) = keras.datasets.cifar100.load_data()
        y = joblib.load(DATA_DIR / 'cifar100_teacher_pred.pkl')
        y_train, y_test = y[:50000, ...], y[50000:, ...]
        logger.info('train, test = %d, %d', len(X_train), len(X_test))

        image_size = (96, 96)
        model = od.create_pretrain_network(image_size)
        logger.info('network depth: %d', tk.dl.count_network_depth(model))
        logger.info('trainable params: %d', tk.dl.count_trainable_params(model))

        # 学習率：
        # ・CIFARなどの分類ではlr 0.5、batch size 256くらいが多いのでその辺を基準に。
        # ・バッチサイズに比例させると良さそう？
        base_lr = 0.5 * (args.batch_size / 256) * hvd.size()

        with tk.log.trace_scope('model.compile'):
            opt = tk.dl.nsgd()(lr=base_lr)
            opt = hvd.DistributedOptimizer(opt)
            model.compile(opt, 'mse', ['mae'])

        gen = tk.image.ImageDataGenerator()
        gen.add(tk.image.RandomFlipLR(probability=0.5))
        gen.add(tk.image.RandomPadding(probability=1))
        gen.add(tk.image.RandomRotate(probability=0.5))
        gen.add(tk.image.RandomCrop(probability=1))
        gen.add(tk.image.Resize(image_size))
        gen.add(tk.image.RandomAugmentors([
            tk.image.RandomBlur(probability=0.5),
            tk.image.RandomUnsharpMask(probability=0.5),
            tk.image.GaussianNoise(probability=0.5),
            tk.image.RandomSaturation(probability=0.5),
            tk.image.RandomBrightness(probability=0.5),
            tk.image.RandomContrast(probability=0.5),
            tk.image.RandomHue(probability=0.5),
        ]))
        gen.add(tk.image.RandomErasing(probability=0.5))
        gen.add(tk.image.ProcessInput(od.get_preprocess_input(), batch_axis=True))

        callbacks = []
        callbacks.append(tk.dl.learning_rate_callback())
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(tk.dl.tsv_log_callback(RESULT_DIR / 'pretrain.history.tsv'))
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
            model.save(str(RESULT_DIR / 'pretrain.model.h5'))


if __name__ == '__main__':
    _main()
