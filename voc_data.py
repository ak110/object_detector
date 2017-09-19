"""データのGenerator。"""
import pathlib

import numpy as np

import pytoolkit as tk

CLASS_NAMES = ['bg'] + tk.ml.VOC_CLASS_NAMES
CLASS_NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def load_data(data_dir: pathlib.Path, debug: bool, batch_size):
    """データの読み込み"""
    X_train = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')) +
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_test = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))
    )
    if debug:
        X_test = X_test[:batch_size]  # 先頭batch_size件だけを使用
        rep = int(np.ceil((len(X_train) / len(X_test)) ** 0.9))  # 個数を減らした分、水増しする。
        X_train = np.tile(X_test, rep)
    annotations = load_annotations(data_dir)
    y_train = np.array([annotations[x] for x in X_train])
    y_test = np.array([annotations[x] for x in X_test])
    # Xのフルパス化
    X_train = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_train])
    X_test = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_test])
    return (X_train, y_train), (X_test, y_test)


def load_annotations(data_dir: pathlib.Path) -> dict:
    """VOC2007,VOC2012のアノテーションデータの読み込み。"""
    data = {}
    for folder in ('VOC2007', 'VOC2012'):
        data.update(tk.ml.ObjectsAnnotation.load_voc(
            data_dir.joinpath('VOCdevkit', folder, 'Annotations'), CLASS_NAME_TO_ID))
    return data
