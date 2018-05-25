#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testの適合率・再現率を算出したり結果をplotしたり。"""
import pathlib

import pytoolkit as tk

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_VOCDEVKIT_DIR = _BASE_DIR / 'data' / 'VOCdevkit'
_RESULTS_DIR = _BASE_DIR / 'results'
_BATCH_SIZE = 16


def _main():
    with tk.dl.session():
        tk.log.init(_RESULTS_DIR / 'evaluate.log')
        _run()


@tk.log.trace()
def _run():
    X_test, y_test = tk.data.voc.load_07_test(_VOCDEVKIT_DIR)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=_BATCH_SIZE, keep_aspect=False, strict_nms=False, use_multi_gpu=True)
    pred = od.predict(X_test, conf_threshold=0.6)

    # 適合率・再現率などを算出・表示
    precisions, recalls, fscores, supports = tk.ml.compute_scores(y_test, pred, iou_threshold=0.5)
    tk.ml.print_scores(precisions, recalls, fscores, supports, tk.data.voc.CLASS_NAMES)

    # 先頭部分のみ可視化
    save_dir = _RESULTS_DIR / '___check'
    for x, p in zip(X_test[:64], pred[:64]):
        img = p.plot(x, tk.data.voc.CLASS_NAMES)
        tk.ndimage.save(save_dir / (x.stem + '.jpg'), img)


if __name__ == '__main__':
    _main()
