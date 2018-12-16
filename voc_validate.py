#!/usr/bin/env python3
"""学習済みモデルでVOC 07 testのmAPを算出。"""
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
        tk.log.init(args.result_dir / 'validate.log')
        _run(args)


@tk.log.trace()
def _run(args):
    X_test, y_test = tk.data.voc.load_07_test(args.vocdevkit_dir)
    od = tk.dl.od.ObjectDetector.load_voc(batch_size=args.batch_size, input_size=args.input_size,
                                          keep_aspect=False, strict_nms=False, use_multi_gpu=True)
    pred_test = od.predict(X_test)

    scores = tk.data.voc.evaluate(y_test, pred_test)
    logger = tk.log.get(__name__)
    logger.info(f'mAP={scores["mAP"] * 100:.1f} mAP(VOC2007)={scores["mAP_VOC"] * 100:.1f}')


if __name__ == '__main__':
    _main()
