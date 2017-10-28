"""ObjectDetectionのモデル。"""
import pathlib
import sys

import joblib
import numpy as np
from tqdm import tqdm

import pytoolkit as tk
from voc_data import CLASS_NAMES


def evaluate(logger, od, model, gen, X_test, y_test, batch_size, epoch, result_dir):
    """`mAP`を算出してprintする。"""
    predict_model = od.create_predict_network(model)

    pred_classes_list = []
    pred_confs_list = []
    pred_locs_list = []
    steps = gen.steps_per_epoch(len(X_test), batch_size)
    with tqdm(total=len(X_test), unit='f', desc='evaluate', ascii=True, ncols=100) as pbar, joblib.Parallel(n_jobs=batch_size, backend='threading') as parallel:
        for i, X_batch in enumerate(gen.flow(X_test, batch_size=batch_size)):
            pred_classes, pred_confs, pred_locs = predict_model.predict(X_batch)
            pred_classes, pred_confs, pred_locs = od.select_predictions(
                pred_classes, pred_confs, pred_locs, parallel=parallel)
            pred_classes_list.extend(pred_classes)
            pred_confs_list.extend(pred_confs)
            pred_locs_list.extend(pred_locs)
            if i == 0:
                save_dir = result_dir.joinpath('___check')
                for j, (pcl, pcf, pl) in enumerate(zip(pred_classes, pred_confs, pred_locs)):
                    mask = pcf >= 0.5  # 表示はある程度絞る (mAP算出のためには片っ端からリストアップしているため)
                    tk.ml.plot_objects(
                        X_test[j], save_dir.joinpath(pathlib.Path(X_test[j]).name + '.png'),
                        pcl[mask], pcf[mask], pl[mask], CLASS_NAMES)
            pbar.update(len(X_batch))
            if i + 1 >= steps:
                break

    gt_classes_list = np.array([y.classes for y in y_test])
    gt_bboxes_list = np.array([y.bboxes for y in y_test])
    gt_difficults_list = np.array([y.difficults for y in y_test])
    map1 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=False)
    map2 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=True)

    sys.stdout.flush()
    sys.stderr.flush()
    logger.debug('epoch={:2d} mAP={:.4f} mAP(VOC2007)={:.4f}'.format(epoch + 1, map1, map2))


def plot_truth(X_test, y_test, save_dir):
    """正解データの画像化。"""
    for X, y in zip(X_test, y_test):
        tk.ml.plot_objects(
            X, save_dir.joinpath(pathlib.Path(X).name + '.png'),
            y.classes, None, y.bboxes, CLASS_NAMES)
