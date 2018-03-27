#!/usr/bin/env python
"""Generatorのお試しコード。"""
import pathlib

import numpy as np

import models
import pytoolkit as tk


def _main():
    import matplotlib
    matplotlib.use('Agg')

    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    save_dir = base_dir / '___generator_check'
    save_dir.mkdir(exist_ok=True)

    class_names = ['bg'] + tk.ml.VOC_CLASS_NAMES
    class_name_to_id = {n: i for i, n in enumerate(class_names)}
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(data_dir, class_name_to_id)
    y_test = y_test[:1]
    X_test = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_test)

    od = models.ObjectDetector('xception', input_size=512, map_sizes=[], nb_classes=100, mean_objets=0, pb_size_patterns=np.array([[1.5, 0.5]]))
    gen = od.create_generator(encode_truth=False)
    for i, (X_batch, y_batch) in zip(tk.tqdm(range(16)), gen.flow(X_test, y_test, data_augmentation=True)):
        for X, y in zip(X_batch, y_batch):
            img = tk.image.unpreprocess_input_abs1(X)
            tk.ml.plot_objects(
                img, save_dir / f'{i}.png',
                y.classes, None, y.bboxes, class_names)


if __name__ == '__main__':
    _main()
