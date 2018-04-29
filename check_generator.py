#!/usr/bin/env python3
"""Generatorのお試しコード。"""
import pathlib

import pytoolkit as tk


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    save_dir = base_dir / '___generator_check'
    save_dir.mkdir(exist_ok=True)

    class_names = ['bg'] + tk.ml.VOC_CLASS_NAMES
    class_name_to_id = {n: i for i, n in enumerate(class_names)}
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(data_dir, class_name_to_id)
    y_test = y_test[:1]
    X_test = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_test)

    gen = tk.dl.od.od_gen.create_generator((512, 512), preprocess_input=lambda x: x, encode_truth=None,
                                           flip_h=True, flip_v=True, rotate90=True)
    for i, (X_batch, y_batch) in zip(tk.tqdm(range(32)), gen.flow(X_test, y_test, data_augmentation=True)):
        for rgb, y in zip(X_batch, y_batch):
            img = tk.ml.plot_objects(rgb, y.classes, None, y.bboxes, class_names)
            tk.ndimage.save(save_dir / f'{i}.jpg', img)


if __name__ == '__main__':
    _main()
