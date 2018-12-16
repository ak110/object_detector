#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testの適合率・再現率を算出したり結果をplotしたり。"""
import argparse
import pathlib

import pytoolkit as tk


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocdevkit-dir', default=pathlib.Path('pytoolkit/data/VOCdevkit'), type=pathlib.Path)
    parser.add_argument('--result-dir', default=pathlib.Path('results'), type=pathlib.Path)
    parser.add_argument('--input-size', default=(320, 320), type=int, nargs=2)
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()
    with tk.dl.session():
        tk.log.init(args.result_dir / 'evaluate.log')
        _run(args)


@tk.log.trace()
def _run(args):
    X_test, y_test = tk.data.voc.load_07_test(args.vocdevkit_dir)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=args.batch_size, input_size=args.input_size,
                                          keep_aspect=False, strict_nms=False, use_multi_gpu=True)
    pred = od.predict(X_test, conf_threshold=0.6)

    # 適合率・再現率などを算出・表示
    precisions, recalls, fscores, supports = tk.ml.compute_scores(y_test, pred, iou_threshold=0.5)
    tk.ml.print_scores(precisions, recalls, fscores, supports, tk.data.voc.CLASS_NAMES)

    # 先頭部分のみ可視化
    save_dir = args.result_dir / '___check'
    for x, p in zip(X_test[:64], pred[:64]):
        img = p.plot(x, tk.data.voc.CLASS_NAMES)
        tk.ndimage.save(save_dir / (x.stem + '.jpg'), img)


if __name__ == '__main__':
    _main()
