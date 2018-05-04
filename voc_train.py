#!/usr/bin/env python3
"""ObjectDetectionを適当にやってみるコード。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / 'data'
_RESULTS_DIR = _BASE_DIR / 'results'
_BASE_NETWORK = 'vgg16'
_INPUT_SIZE = (320, 320)
_MAP_SIZES = [40, 20, 10]
_PB_SIZES = 8
_BATCH_SIZE = 16
_EPOCHS = 300
_HOROVOD = True


def _main():
    with tk.dl.session(use_horovod=_HOROVOD):
        tk.log.init(_RESULTS_DIR / 'train.log')
        _run()


@tk.log.trace()
def _run():
    # データの読み込み
    X_train, y_train = tk.ml.ObjectsAnnotation.load_voc_0712_trainval(_DATA_DIR, without_difficult=True)
    X_val, y_val = tk.ml.ObjectsAnnotation.load_voc_07_test(_DATA_DIR)
    num_classes = len(tk.ml.VOC_CLASS_NAMES)

    # 重みがあれば読み込む
    weights = 'imagenet'
    for warm_path in (_RESULTS_DIR / 'model.base.h5', _RESULTS_DIR / 'pretrain.model.h5'):
        if warm_path.is_file():
            weights = warm_path
            break

    # 学習
    od = tk.dl.od.ObjectDetector(_BASE_NETWORK, _INPUT_SIZE, _MAP_SIZES, num_classes)
    od.fit(X_train, y_train, X_val, y_val,
           batch_size=_BATCH_SIZE, epochs=_EPOCHS, initial_weights=weights, pb_size_pattern_count=_PB_SIZES,
           flip_h=True, flip_v=False, rotate90=False,
           plot_path=_RESULTS_DIR / 'model.svg',
           tsv_log_path=_RESULTS_DIR / 'train.history.tsv')
    # 保存
    od.save(_RESULTS_DIR / 'model.json')
    od.save_weights(_RESULTS_DIR / 'model.h5')


if __name__ == '__main__':
    _main()
