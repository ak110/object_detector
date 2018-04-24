#!/usr/bin/env python
"""ObjectDetectionを適当にやってみるコード。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / 'data'
_RESULTS_DIR = _BASE_DIR / 'results'
_BASE_NETWORK = 'vgg16'
_INPUT_SIZE = (320, 320)
_MAP_SIZES = [40, 20, 10]
_BATCH_SIZE = 16
_EPOCHS = 5
_HOROVOD = True


def _main():
    with tk.dl.session(use_horovod=_HOROVOD):
        tk.log.init(_RESULTS_DIR / 'train.log')
        _run()


@tk.log.trace()
def _run():
    # データの読み込み
    y_train = tk.ml.ObjectsAnnotation.load_voc_0712_trainval(_DATA_DIR, without_difficult=True)
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(_DATA_DIR)
    X_train = tk.ml.ObjectsAnnotation.get_path_list(_DATA_DIR, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(_DATA_DIR, y_test)
    num_classes = len(tk.ml.VOC_CLASS_NAMES)

    # 重みがあれば読み込む
    weights = 'imagenet'
    for warm_path in (_RESULTS_DIR / 'model.base.h5', _RESULTS_DIR / 'pretrain.model.h5'):
        if warm_path.is_file():
            weights = warm_path
            break

    # 準備
    od = tk.dl.od.ObjectDetector(_BASE_NETWORK, _INPUT_SIZE, _MAP_SIZES, num_classes)
    od.fit(y_train, y_test, batch_size=_BATCH_SIZE, initial_weights=weights)
    od.save(_RESULTS_DIR / 'model.json')
    od.model.plot(_RESULTS_DIR / 'model.svg')

    # 学習
    callbacks = []
    callbacks.append(tk.dl.callbacks.learning_rate(reduce_epoch_rates=(0.5, 0.75, 0.875)))
    callbacks.extend(od.model.horovod_callbacks())
    callbacks.append(tk.dl.callbacks.tsv_logger(_RESULTS_DIR / 'train.history.tsv'))
    callbacks.append(tk.dl.callbacks.epoch_logger())
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))
    od.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=_EPOCHS, callbacks=callbacks)
    # 保存
    od.save_weights(_RESULTS_DIR / 'model.h5')


if __name__ == '__main__':
    _main()
