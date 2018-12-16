#!/usr/bin/env python3
"""Generatorのお試しコード。"""
import pathlib

import numpy as np

import pytoolkit as tk


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    vocdevkit_dir = base_dir / 'pytoolkit' / 'data' / 'VOCdevkit'
    save_dir = base_dir / '___assign_check'
    save_dir.mkdir(exist_ok=True)

    X_val, y_val = tk.data.voc.load_07_test(vocdevkit_dir)
    X_val, y_val = X_val[:1], y_val[:1]

    od = tk.dl.od.ObjectDetector.load_voc(batch_size=1, use_multi_gpu=False)

    gen = tk.dl.od.od_gen.create_generator((512, 512), preprocess_input=lambda x: x, encode_truth=od.pb.encode_truth)
    g, _ = gen.flow(X_val, y_val, data_augmentation=True)
    for i, (X_batch, y_batch) in zip(tk.tqdm(range(32)), g):
        for rgb, y in zip(X_batch, y_batch):
            obj_pb = y[:, 1] == 1  # assignされたもの全部
            classes = np.argmax(y[obj_pb, 2:-4], axis=-1)
            bboxes = od.pb.decode_locs(np.zeros((len(y), 4)), xp=np)[obj_pb, :]  # prior box自体の座標を取る

            img = tk.ml.plot_objects(rgb, classes, None, bboxes, tk.data.voc.CLASS_NAMES)
            tk.ndimage.save(save_dir / f'{i}.jpg', img)


if __name__ == '__main__':
    _main()
