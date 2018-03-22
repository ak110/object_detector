#!/usr/bin/env python
"""処理結果の可視化などを行うスクリプト。"""
import argparse
import pathlib

import pandas as pd

import config
import models
import pytoolkit as tk


def _main():
    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='動作種別', nargs='?', default='all', choices=['net', 'history', 'all'])
    parser.add_argument('--result-dir', help='resultsディレクトリのパス。', default=str(config.RESULT_DIR))
    args = parser.parse_args()
    config.RESULT_DIR = pathlib.Path(args.result_dir)

    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler(level='DEBUG'))

    if args.mode == 'net':
        _report_net()
    elif args.mode == 'history':
        _report_history()
    else:
        _report_net()
        _report_history()


def _report_net():
    od = models.ObjectDetector.load(config.RESULT_DIR / 'model.pkl')

    import keras
    with tk.dl.session():
        model, _ = od.create_network()

        with tk.log.trace_scope('network.txt'):
            with (config.RESULT_DIR / 'network.txt').open('w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

        with tk.log.trace_scope('model.params.png'):
            tk.dl.plot_model_params(model, config.RESULT_DIR / 'model.params.png')

        with tk.log.trace_scope('model.png'):
            keras.utils.plot_model(model, str(config.RESULT_DIR / 'model.png'), show_shapes=False)


def _report_history():
    import matplotlib.pyplot as plt

    with tk.log.trace_scope('history.loss.png'):
        df = pd.read_csv(str(config.RESULT_DIR / 'history.tsv'), sep='\t')
        df[['loss', 'val_loss']].plot()
        plt.savefig(str(config.RESULT_DIR / 'history.loss.png'))
        plt.close()


if __name__ == '__main__':
    _main()
