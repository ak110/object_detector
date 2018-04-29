#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import pathlib

import numpy as np

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
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(_DATA_DIR)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(_DATA_DIR, y_test)
    class_names = tk.ml.VOC_CLASS_NAMES

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load(_RESULTS_DIR / 'model.json')
    od.load_weights(_RESULTS_DIR / 'model.h5', batch_size=_BATCH_SIZE, strict_nms=False, use_multi_gpu=True)

    # 予測
    pred_classes_list, pred_confs_list, pred_locs_list = od.predict(X_test)

    # mAP
    gt_classes_list = np.array([y.classes for y in y_test])
    gt_bboxes_list = np.array([y.bboxes for y in y_test])
    gt_difficults_list = np.array([y.difficults for y in y_test])
    map1 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=False)
    map2 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=True)
    logger.info(f'mAP={map1:.4f} mAP(VOC2007)={map2:.4f}')

    # 先頭部分のみ可視化
    save_dir = _RESULTS_DIR / '___check'
    for x, pcl, pcf, pl in zip(X_test[:64], pred_classes_list[:64], pred_confs_list[:64], pred_locs_list[:64]):
        mask = pcf >= 0.5  # 表示はある程度絞る (mAP算出のためには片っ端からリストアップしているため)
        if not mask.any():
            mask = [pcf.argmax()]  # 最低1つは出力
        tk.ml.plot_objects(x, save_dir / (x.name + '.png'), pcl[mask], pcf[mask], pl[mask], class_names)


if __name__ == '__main__':
    _main()