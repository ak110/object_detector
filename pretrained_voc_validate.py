#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testのmAPを算出。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / 'data'
_RESULTS_DIR = _BASE_DIR / 'results'
_BATCH_SIZE = 16


def _main():
    with tk.dl.session():
        tk.log.init(_RESULTS_DIR / 'validate.log')
        _run()


@tk.log.trace()
def _run():
    logger = tk.log.get(__name__)

    # データの読み込み
    X_test, y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(_DATA_DIR)

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=_BATCH_SIZE, strict_nms=False, use_multi_gpu=True)

    # 予測
    pred_test = od.predict(X_test)

    # mAP
    map1 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=True)
    logger.info(f'mAP={map1:.4f} mAP(VOC2007)={map2:.4f}')


if __name__ == '__main__':
    _main()
