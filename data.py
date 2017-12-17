"""データのGenerator。"""
import pathlib

import numpy as np
import sklearn.externals.joblib

import pytoolkit as tk


def load_data(data_dir: pathlib.Path, data_type: str):
    """データの読み込み"""
    assert data_type in ('voc', 'pkl')
    if data_type == 'voc':
        return load_data_voc(data_dir)
    else:
        return load_data_pkl(data_dir)


def load_data_voc(data_dir: pathlib.Path):
    """データの読み込み"""
    class_names = ['bg'] + tk.ml.VOC_CLASS_NAMES
    class_name_to_id = {n: i for i, n in enumerate(class_names)}
    X_train1 = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_train2 = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'))
    )
    X_test = np.array(
        tk.io.read_all_lines(data_dir.joinpath('VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt'))
    )
    X_train = np.concatenate([X_train1, X_train2], axis=0)
    y_train = np.concatenate([
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2007', 'Annotations'), X_train1, class_name_to_id, without_difficult=True),
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2012', 'Annotations'), X_train2, class_name_to_id, without_difficult=True)
    ], axis=0)
    y_test = np.array(
        tk.ml.ObjectsAnnotation.load_voc(data_dir.joinpath('VOCdevkit', 'VOC2007', 'Annotations'), X_test, class_name_to_id)
    )
    # Xのフルパス化
    X_train = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_train])
    X_test = np.array([data_dir.joinpath('VOCdevkit', y.folder, 'JPEGImages', y.filename) for y in y_test])
    return (X_train, y_train), (X_test, y_test), class_names


def load_data_pkl(data_dir: pathlib.Path):
    """データの読み込み"""
    y_train = sklearn.externals.joblib.load(data_dir / 'y_train.pkl')
    y_test = sklearn.externals.joblib.load(data_dir / 'y_test.pkl')
    class_names = sklearn.externals.joblib.load(data_dir / 'class_names.pkl')
    X_train = np.array([data_dir / y.folder / y.filename for y in y_train])
    X_test = np.array([data_dir / y.folder / y.filename for y in y_test])
    return (X_train, y_train), (X_test, y_test), class_names
