#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testのmAPを算出。"""
import argparse
import pathlib

import pytoolkit as tk


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocdevkit-dir', default=pathlib.Path('data/VOCdevkit'), type=pathlib.Path)
    parser.add_argument('--result-dir', default=pathlib.Path('results'), type=pathlib.Path)
    parser.add_argument('--input-size', default=(320, 320), type=int, nargs=2)
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()
    with tk.dl.session():
        tk.log.init(args.result_dir / 'validate.log')
        _run(args)


@tk.log.trace()
def _run(args):
    X_test, y_test = tk.data.voc.load_07_test(args.vocdevkit_dir)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=args.batch_size, input_size=args.input_size,
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
        gt_bboxes_list = np.array([y.real_bboxes for y in y_test])
        gt_difficults_list = np.array([y.difficults for y in y_test])
        pred_classes_list = np.array([p.classes for p in pred_test])
        pred_confs_list = np.array([p.confs for p in pred_test])
        pred_bboxes_list = np.array([p.get_real_bboxes(y.width, y.height) for (p, y) in zip(pred_test, y_test)])
        scores1 = chainercv.evaluations.eval_detection_voc(
            pred_bboxes_list, pred_classes_list, pred_confs_list,
            gt_bboxes_list, gt_classes_list, gt_difficults_list,
            use_07_metric=False)
        scores2 = chainercv.evaluations.eval_detection_voc(
            pred_bboxes_list, pred_classes_list, pred_confs_list,
            gt_bboxes_list, gt_classes_list, gt_difficults_list,
            use_07_metric=True)
        logger.info(f'ChainerCV evaluation: mAP={scores1["map"]:.3f} mAP(VOC2007)={scores2["map"]:.3f}')
    except BaseException:
        logger.warning(f'ChainerCV error', exc_info=True)


if __name__ == '__main__':
    _main()
