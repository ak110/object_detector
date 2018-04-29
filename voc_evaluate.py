#!/usr/bin/env python3
"""学習済みモデルで検証してみるコード。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / 'data'
_RESULTS_DIR = _BASE_DIR / 'results'
_BATCH_SIZE = 16


def _main():
    with tk.dl.session():
        tk.log.init(_RESULTS_DIR / 'evaluate.log')
        _run()


@tk.log.trace()
def _run():
    # データの読み込み
    X_test, y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(_DATA_DIR)
    class_names = tk.ml.VOC_CLASS_NAMES

    # モデルの読み込み
    od = tk.dl.od.ObjectDetector.load(_RESULTS_DIR / 'model.json')
    od.load_weights(_RESULTS_DIR / 'model.h5', batch_size=_BATCH_SIZE, strict_nms=True, use_multi_gpu=True)

    # 予測
    pred = od.predict(X_test, conf_threshold=0.75)

    # 適合率・再現率などを算出・表示
    precisions, recalls, fscores, supports = tk.ml.compute_scores(y_test, pred, iou_threshold=0.5)
    tk.ml.print_scores(precisions, recalls, fscores, supports, class_names)

    # 先頭部分のみ可視化
    save_dir = _RESULTS_DIR / '___check'
    for x, p in zip(X_test[:64], pred[:64]):
        tk.ndimage.save(save_dir / x.with_suffix('.jpg').name, p.plot(x, class_names))


if __name__ == '__main__':
    _main()
