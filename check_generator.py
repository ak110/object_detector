#!/usr/bin/env python3
"""Generatorのお試しコード。"""
import pathlib

import pytoolkit as tk


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    vocdevkit_dir = base_dir / 'data' / 'VOCdevkit'
    save_dir = base_dir / '___generator_check'
    save_dir.mkdir(exist_ok=True)

    X_val, y_val = tk.data.voc.load_07_test(vocdevkit_dir)
    X_val, y_val = X_val[:1], y_val[:1]

    gen = tk.dl.od.od_gen.create_generator((512, 512), preprocess_input=lambda x: x, encode_truth=None)
    g, _ = gen.flow(X_val, y_val, data_augmentation=True)
    for i, (X_batch, y_batch) in zip(tk.tqdm(range(32)), g):
        for rgb, y in zip(X_batch, y_batch):
            img = tk.ml.plot_objects(rgb, y.classes, None, y.bboxes, tk.data.voc.CLASS_NAMES)
            tk.ndimage.save(save_dir / f'{i}.jpg', img)


if __name__ == '__main__':
    _main()
