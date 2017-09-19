"""ObjectDetectionを適当にやってみるコード。

-train: VOC2007 trainval + VOC2012 trainval
-test:  VOC2007 test
"""
import os
import pathlib
import time

import better_exceptions
import numpy as np

import pytoolkit as tk
from evaluation import evaluate, plot_truth
from generator import Generator
from model import ObjectDetector
from model_net import create_network
from voc_data import CLASS_NAMES, load_data

_DEBUG = True

_BATCH_SIZE = 16
_MAX_EPOCH = 16 if _DEBUG else 300
_BASE_LR = 1e-1  # 1e-3

_CHECK_PRIOR_BOX = True  # 重いので必要なときだけ
_EVALUATE = True  # 重いので必要なときだけ


def _main():
    import matplotlib as mpl
    mpl.use('Agg')

    better_exceptions.MAX_LENGTH = 128

    script_path = pathlib.Path(__file__).resolve()
    base_dir = script_path.parent
    os.chdir(str(base_dir))
    np.random.seed(1337)  # for reproducibility

    result_dir = base_dir.joinpath('results', script_path.stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = tk.create_tee_logger(result_dir.joinpath(script_path.stem + '.log'))

    start_time = time.time()
    _run(logger, result_dir, base_dir.joinpath('data'))
    elapsed_time = time.time() - start_time

    logger.info('Elapsed time = %d [s]', int(np.ceil(elapsed_time)))


def _run(logger, result_dir: pathlib.Path, data_dir: pathlib.Path):
    # データの読み込み
    (X_train, y_train), (X_test, y_test) = load_data(data_dir, _DEBUG, _BATCH_SIZE)
    logger.debug('train, test = %d, %d', len(X_train), len(X_test))

    # 試しに回答を出力してみる。
    plot_truth(X_test[:_BATCH_SIZE], y_test[:_BATCH_SIZE], result_dir.joinpath('___ground_truth'))
    # for i, y in enumerate(y_test[:3]):
    #     logger.debug('y_test[%d]: %s', i, str(y))

    # 訓練データからパラメータを適当に決める。
    # gridに配置したときのIOUを直接最適化するのは難しそうなので、
    # とりあえず大雑把にKMeansでクラスタ化したりなど。
    od = ObjectDetector.create(len(CLASS_NAMES), y_train)
    logger.debug('mean objects / image = %f', od.mean_objets)
    logger.debug('prior box size ratios = %s', str(od.pb_size_ratios))
    logger.debug('prior box aspect ratios = %s', str(od.aspect_ratios))
    logger.debug('prior box count = %d', len(od.pb_locs))
    logger.debug('prior box sizes = %s', str(np.unique(od.pb_sizes)))

    # prior boxのカバー度合いのチェック
    if _DEBUG or _CHECK_PRIOR_BOX:
        od.check_prior_boxes(logger, result_dir, y_test, CLASS_NAMES)

    import keras
    import keras.backend as K
    K.set_image_dim_ordering('tf')
    with tk.dl.session():
        model = create_network(od)
        model.summary(print_fn=logger.debug)
        tk.dl.plot_model_params(model, result_dir.joinpath('model.params.png'))
        keras.utils.plot_model(model, str(result_dir.joinpath('model.png')), show_shapes=True)

        # 事前学習の読み込み
        # model.load_weights(str(result_dir.parent.joinpath('voc_pre', 'model.h5')), by_name=True)

        gen = Generator(image_size=od.input_size, od=od)
        gen.add(0.5, tk.image.FlipLR())
        gen.add(0.125, tk.image.RandomBlur())
        gen.add(0.125, tk.image.RandomBlur(partial=True))
        gen.add(0.125, tk.image.RandomUnsharpMask())
        gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
        gen.add(0.125, tk.image.Sharp())
        gen.add(0.125, tk.image.Sharp(partial=True))
        gen.add(0.125, tk.image.Soft())
        gen.add(0.125, tk.image.Soft(partial=True))
        gen.add(0.125, tk.image.RandomMedian())
        gen.add(0.125, tk.image.RandomMedian(partial=True))
        gen.add(0.125, tk.image.GaussianNoise())
        gen.add(0.125, tk.image.GaussianNoise(partial=True))
        gen.add(0.125, tk.image.RandomSaturation())
        gen.add(0.125, tk.image.RandomBrightness())
        gen.add(0.125, tk.image.RandomContrast())
        gen.add(0.125, tk.image.RandomLighting())

        # lr_list = [_BASE_LR] * (_MAX_EPOCH * 6 // 9) + [_BASE_LR / 10] * (_MAX_EPOCH * 2 // 9) + [_BASE_LR / 100] * (_MAX_EPOCH * 1 // 9)

        callbacks = []
        # callbacks.append(tk.dl.my_callback_factory()(result_dir, lr_list=lr_list))
        callbacks.append(tk.dl.my_callback_factory()(result_dir, base_lr=_BASE_LR, beta1=0.9990, beta2=0.9995))
        callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'loss'))
        callbacks.append(keras.callbacks.ModelCheckpoint(str(result_dir.joinpath('model.best.h5')), save_best_only=True))
        # callbacks.append(tk.dl.learning_curve_plotter_factory()(result_dir.joinpath('history.{metric}.png'), 'acc'))
        # if K.backend() == 'tensorflow':
        #     callbacks.append(keras.callbacks.TensorBoard())

        # 各epoch毎にmAPを算出して表示してみる
        if _DEBUG or _EVALUATE:
            callbacks.append(keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: evaluate(logger, od, model, gen, X_test, y_test, _BATCH_SIZE, epoch, result_dir)
            ))

        model.fit_generator(
            gen.flow(X_train, y_train, batch_size=_BATCH_SIZE, data_augmentation=not _DEBUG, shuffle=not _DEBUG),
            steps_per_epoch=gen.steps_per_epoch(len(X_train), _BATCH_SIZE),
            epochs=_MAX_EPOCH,
            validation_data=gen.flow(X_test, y_test, batch_size=_BATCH_SIZE),
            validation_steps=gen.steps_per_epoch(len(X_test), _BATCH_SIZE),
            callbacks=callbacks)

        model.save(str(result_dir.joinpath('model.h5')))

        # 最終結果表示
        evaluate(logger, od, model, gen, X_test, y_test, _BATCH_SIZE, None, result_dir)


if __name__ == '__main__':
    _main()
