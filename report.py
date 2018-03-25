#!/usr/bin/env python
"""処理結果の可視化などを行うスクリプト。"""
import argparse
import pathlib

import pandas as pd

import models
import pytoolkit as tk

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='動作種別', nargs='?', default='all', choices=['net', 'history', 'all'])
    parser.add_argument('--result-dir', help='resultsディレクトリのパス。', type=pathlib.Path, default=BASE_DIR / 'results')
    args = parser.parse_args()
    tk.log.init(None, stream_level='DEBUG')
    if args.mode == 'net':
        _report_net(args)
    elif args.mode == 'history':
        _report_history(args)
    else:
        _report_net(args)
        _report_history(args)


def _report_net(args):
    od = models.ObjectDetector.load(args.result_dir / 'model.pkl')

    import keras
    with tk.dl.session():
        model, _ = od.create_network()

        with tk.log.trace_scope('network.txt'):
            with (args.result_dir / 'network.txt').open('w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

        with tk.log.trace_scope('model.params.png'):
            tk.dl.models.plot_model_params(model, args.result_dir / 'model.params.png')

        with tk.log.trace_scope('model.png'):
            try:
                keras.utils.plot_model(model, str(args.result_dir / 'model.png'), show_shapes=False)
            except BaseException:
                tk.log.get(__name__).warning('keras.utils.plot_model失敗', exc_info=True)


def _report_history(args):
    import matplotlib.pyplot as plt

    with tk.log.trace_scope('history.loss.png'):
        df = pd.read_csv(str(args.result_dir / 'history.tsv'), sep='\t')
        df[['loss', 'val_loss']].plot()
        plt.savefig(str(args.result_dir / 'history.loss.png'))
        plt.close()


if __name__ == '__main__':
    _main()
