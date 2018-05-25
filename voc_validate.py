#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testのmAPを算出。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_VOCDEVKIT_DIR = _BASE_DIR / 'data' / 'VOCdevkit'
_RESULTS_DIR = _BASE_DIR / 'results'
_BATCH_SIZE = 16


def _main():
    with tk.dl.session():
        tk.log.init(_RESULTS_DIR / 'validate.log')
        _run()


@tk.log.trace()
def _run():
    X_test, y_test = tk.data.voc.load_07_test(_VOCDEVKIT_DIR)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=_BATCH_SIZE, keep_aspect=False, strict_nms=False, use_multi_gpu=True)
    pred_test = od.predict(X_test)

    # mAP
    map1 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=True)
    logger = tk.log.get(__name__)
    logger.info(f'mAP={map1:.3f} mAP(VOC2007)={map2:.3f}')


if __name__ == '__main__':
    _main()
