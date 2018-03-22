#!/usr/bin/env python
"""学習済みモデルで検証してみるコード。"""
import argparse
import pathlib

import better_exceptions
import numpy as np
import sklearn.externals.joblib as joblib
from tqdm import tqdm

import data
import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'


def _main():
    better_exceptions.MAX_LENGTH = 100
    import matplotlib as mpl
    mpl.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', help='データの種類。', default='pkl', choices=['voc', 'csv', 'pkl'])
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    args = parser.parse_args()

    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(RESULT_DIR / (pathlib.Path(__file__).stem + '.log')))
    _run(logger, args)


@tk.log.trace()
def _run(logger, args):
    # データの読み込み
    (X_train, _), (X_test, y_test), class_names = data.load_data(DATA_DIR, args.data_type)
    logger.info('train, test = %d, %d', len(X_train), len(X_test))

    with tk.dl.session():
        # モデルの読み込み
        od = models.ObjectDetector.load(RESULT_DIR / 'model.pkl')
        model, _ = od.create_network()
        model.load_weights(str(RESULT_DIR / 'model.h5'))
        # マルチGPU対応
        logger.info('gpu count = %d', tk.get_gpu_count())
        model, batch_size = tk.dl.create_data_parallel_model(model, args.batch_size)

        # 評価
        _evaluate(logger, od, model, X_test, y_test, batch_size, -1, class_names, RESULT_DIR)


def _evaluate(logger, od, model, X_test, y_test, batch_size, epoch, class_names, result_dir):
    """`mAP`を算出してprintする。"""
    predict_model = od.create_predict_network(model)
    gen = od.create_generator()

    pred_classes_list = []
    pred_confs_list = []
    pred_locs_list = []
    steps = gen.steps_per_epoch(len(X_test), batch_size)
    with tqdm(total=len(X_test), unit='f', desc='evaluate', ascii=True, ncols=100) as pbar, joblib.Parallel(n_jobs=batch_size, backend='threading') as parallel:
        for i, X_batch in enumerate(gen.flow(X_test, batch_size=batch_size)):
            # 予測
            pred_classes, pred_confs, pred_locs = predict_model.predict(X_batch)
            # NMSなど
            pred_classes, pred_confs, pred_locs = od.select_predictions(
                pred_classes, pred_confs, pred_locs, parallel=parallel)
            pred_classes_list.extend(pred_classes)
            pred_confs_list.extend(pred_confs)
            pred_locs_list.extend(pred_locs)
            # prior box毎の再現率などの調査
            # TODO: assignでmaxのみ取ってきたい
            # 先頭部分のみ可視化
            if i == 0:
                save_dir = result_dir / '___check'
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

    gt_classes_list = np.array([y.classes for y in y_test])
    gt_bboxes_list = np.array([y.bboxes for y in y_test])
    gt_difficults_list = np.array([y.difficults for y in y_test])
    map1 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=False)
    map2 = tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                             pred_classes_list, pred_confs_list, pred_locs_list,
                             use_voc2007_metric=True)

    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info('epoch={:2d} mAP={:.4f} mAP(VOC2007)={:.4f}'.format(epoch + 1, map1, map2))


if __name__ == '__main__':
    _main()
