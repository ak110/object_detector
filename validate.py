#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import models
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
    od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')
    model = od.create_network(load_weights=False, for_predict=True)
    model.load_weights(str(RESULT_DIR / 'model.h5'), by_name=True)

    # マルチGPU対応
    logger.info('gpu count = %d', tk.get_gpu_count())
    model, batch_size = tk.dl.models.multi_gpu_model(model, args.batch_size)

    # 評価
    gen = od.create_generator()
    pred_classes_list = []
    pred_confs_list = []
    pred_locs_list = []
    steps = gen.steps_per_epoch(len(X_test), batch_size)
    with tk.tqdm(total=len(X_test), unit='f', desc='evaluate') as pbar, joblib.Parallel(batch_size, backend='threading') as parallel:
        for i, X_batch in enumerate(gen.flow(X_test, batch_size=batch_size)):
            # 予測
            pred_classes, pred_confs, pred_locs = model.predict(X_batch)
            # NMSなど
            pred_classes, pred_confs, pred_locs = od.select_predictions(
                pred_classes, pred_confs, pred_locs, parallel=parallel)
            pred_classes_list.extend(pred_classes)
            pred_confs_list.extend(pred_confs)
            pred_locs_list.extend(pred_locs)
            # 先頭部分のみ可視化
            if i == 0:
                save_dir = RESULT_DIR / '___check'
                for j, (pcl, pcf, pl) in enumerate(zip(pred_classes, pred_confs, pred_locs)):
                    mask = pcf >= 0.5  # 表示はある程度絞る (mAP算出のためには片っ端からリストアップしているため)
                    if not mask.any():
                        mask = [pcf.argmax()]  # 最低1つは出力
                    tk.ml.plot_objects(
                        X_test[j], save_dir / (pathlib.Path(X_test[j]).name + '.png'),
                        pcl[mask], pcf[mask], pl[mask], class_names)
            # 次へ
            pbar.update(len(X_batch))
            if i + 1 >= steps:
                break

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


if __name__ == '__main__':
    _main()
