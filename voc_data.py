"""データのGenerator。"""
import pathlib

import numpy as np

import pytoolkit as tk

CLASS_NAMES = ['bg'] + tk.ml.VOC_CLASS_NAMES
CLASS_NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def load_data(data_dir: pathlib.Path, debug: bool, batch_size):
    """データの読み込み"""
    X_train1 = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_train2 = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_test = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))
    )
    if debug:
        X_train1 = X_test[:batch_size]  # 先頭batch_size件だけを使用
        X_train2 = []
        X_test = X_train1
    X_train = np.concatenate([X_train1, X_train2], axis=0)
    y_train = np.concatenate([
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2007', 'Annotations'), X_train1, CLASS_NAME_TO_ID, without_difficult=True),
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2012', 'Annotations'), X_train2, CLASS_NAME_TO_ID, without_difficult=True)
    ], axis=0)
    y_test = np.array(
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2007', 'Annotations'), X_test, CLASS_NAME_TO_ID)
    )
    # Xのフルパス化
    X_train = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_train])
    X_test = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_test])
    return (X_train, y_train), (X_test, y_test)
