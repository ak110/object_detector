#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    args = parser.parse_args()
    tk.log.init(RESULT_DIR / (pathlib.Path(__file__).stem + '.log'))
    with tk.dl.session():
        _run(args)


@tk.log.trace()
def _run(args):
    logger = tk.log.get(__name__)

    # データの読み込み
    _, y_test, class_names = joblib.load(DATA_DIR / 'train_data.pkl')
    X_test = tk.ml.ObjectsAnnotation.get_path_list(DATA_DIR, y_test)

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model = od.create_model('predict', args.batch_size, weights=RESULT_DIR / 'model.h5')

    # 予測
    pred_classes_list, pred_confs_list, pred_locs_list = od.predict(model, X_test)

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
    save_dir = RESULT_DIR / '___check'
    for x, pcl, pcf, pl in zip(X_test[:64], pred_classes_list[:64], pred_confs_list[:64], pred_locs_list[:64]):
        mask = pcf >= 0.5  # 表示はある程度絞る (mAP算出のためには片っ端からリストアップしているため)
        if not mask.any():
            mask = [pcf.argmax()]  # 最低1つは出力
        tk.ml.plot_objects(x, save_dir / (x.name + '.png'), pcl[mask], pcf[mask], pl[mask], class_names)


if __name__ == '__main__':
    _main()
