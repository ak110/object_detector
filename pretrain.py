#!/usr/bin/env python
"""事前学習。"""
import os
import pathlib
import time

import better_exceptions
import numpy as np
import sklearn.externals.joblib

import pytoolkit as tk
from model import ObjectDetector

_BATCH_SIZE = 25
_MAX_EPOCH = 100
_BASE_LR = 1e-3


def _main():
    import matplotlib as mpl
    mpl.use('Agg')
    better_exceptions.MAX_LENGTH = 128
    script_path = pathlib.Path(__file__).resolve()
    base_dir = script_path.parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    result_dir = base_dir.joinpath('results_pretrain')
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath(script_path.stem + '.log'))

    start_time = time.time()
    _run(logger, result_dir)
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, result_dir: pathlib.Path):
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        _run2(logger, result_dir)


def _run2(logger, result_dir: pathlib.Path):
    # データの読み込み
    import keras
    input_shape = (256, 256, 3)
    nb_classes = 100
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    model = ObjectDetector.create_pretrain_network(input_shape, nb_classes)
    model.summary(print_fn=logger.debug)
    logger.debug('network depth: %d', tk.dl.count_network_depth(model))
    tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))
    keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)

    # マルチGPU対応
    gpu_count = tk.get_gpu_count()
    logger.debug('gpu count = %d', gpu_count)
    if gpu_count <= 1:
        batch_size = _BATCH_SIZE
    else:
        model = tk.dl.create_data_parallel_model(model, gpu_count)
        batch_size = _BATCH_SIZE * gpu_count
        keras.utils.plot_model(model, str(result_dir.joinpath('model.multigpu.png')), show_shapes=True)

    model.compile('nadam', 'categorical_crossentropy', ['acc'])

    gen = tk.image.ImageDataGenerator(input_shape[:2])
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.Sharp())
    gen.add(0.125, tk.image.Soft())
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomLighting())

    callbacks = []
    callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=_BASE_LR, max_reduces=0))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
    callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=batch_size, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(len(X_train), batch_size),
        epochs=_MAX_EPOCH,
        validation_data=gen.flow(X_test, y_test, batch_size=batch_size),
        validation_steps=gen.steps_per_epoch(len(X_test), batch_size),
        callbacks=callbacks)

    model.save(str(result_dir.joinpath('model.h5')))

    score = model.evaluate_generator(
        gen.flow(X_test, y_test, batch_size=batch_size),
        gen.steps_per_epoch(X_test.shape[0], batch_size))
    logger.info('Test loss:     {}'.format(score[0]))
    logger.info('Test accuracy: {}'.format(score[1]))
    logger.info('Test error:    {}'.format(1 - score[1]))

    pred = model.predict_generator(
        gen.flow(X_test, batch_size=batch_size),
        gen.steps_per_epoch(X_test.shape[0], batch_size))
    cm = sklearn.metrics.confusion_matrix(y_test.argmax(axis=-1), pred.argmax(axis=-1))
    tk.ml.plot_cm(cm, result_dir.joinpath('confusion_matrix.png'))


if __name__ == '__main__':
    _main()
