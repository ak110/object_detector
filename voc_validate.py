#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testのmAPを算出。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_VOCDEVKIT_DIR = _BASE_DIR / 'data' / 'VOCdevkit'
_RESULTS_DIR = _BASE_DIR / 'results'
_INPUT_SIZE = (640, 640)
_BATCH_SIZE = 16


def _main():
    with tk.dl.session():
        tk.log.init(_RESULTS_DIR / 'validate.log')
        _run()


@tk.log.trace()
def _run():
    X_test, y_test = tk.data.voc.load_07_test(_VOCDEVKIT_DIR)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=_BATCH_SIZE, input_size=_INPUT_SIZE,
                                          keep_aspect=False, strict_nms=False, use_multi_gpu=True)
    pred_test = od.predict(X_test)

    # mAP
    map1 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(y_test, pred_test, use_voc2007_metric=True)
    logger = tk.log.get(__name__)
    logger.info(f'mAP={map1:.3f} mAP(VOC2007)={map2:.3f}')

    # 検算のつもり
    try:
        import chainercv
        import numpy as np
        gt_classes_list = np.array([y.classes for y in y_test])
        gt_bboxes_list = np.array([y.bboxes for y in y_test])
        gt_difficults_list = np.array([y.difficults for y in y_test])
        pred_classes_list = np.array([p.classes for p in pred_test])
        pred_confs_list = np.array([p.confs for p in pred_test])
        pred_bboxes_list = np.array([p.bboxes for p in pred_test])
        scores = chainercv.evaluations.eval_detection_voc(
            pred_bboxes_list, pred_classes_list, pred_confs_list,
            gt_bboxes_list, gt_classes_list, gt_difficults_list,
            use_07_metric=True)
        logger.info(f'voc mAP: {scores["map"] * 100:.1f}')
    except BaseException:
        logger.warning(f'ChainerCV error', exc_info=True)


if __name__ == '__main__':
    _main()
