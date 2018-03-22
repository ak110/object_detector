"""データの読み込み。"""
import pathlib
import warnings

import numpy as np
import pandas as pd
import PIL.Image
import sklearn.externals.joblib as joblib
from tqdm import tqdm

import pytoolkit as tk


@tk.log.trace()
def load_data(data_dir, data_type: str):
    """データの読み込み"""
    assert data_type in ('voc', 'csv', 'pkl')
    data_dir = pathlib.Path(data_dir)

    if data_type == 'voc':
        y_train, y_test, class_names = load_data_voc(data_dir)
    elif data_type == 'csv':
        y_train, y_test, class_names = load_data_csv(data_dir)
    else:
        y_train, y_test, class_names = load_data_pkl(data_dir)

    X_train = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_train)
    X_test = tk.ml.ObjectsAnnotation.get_path_list(data_dir, y_test)

    logger = tk.log.get(__name__)
    logger.info('train, test = %d, %d', len(X_train), len(X_test))

    return (X_train, y_train), (X_test, y_test), class_names


def load_data_voc(data_dir: pathlib.Path):
    """データの読み込み"""
    class_names = ['bg'] + tk.ml.VOC_CLASS_NAMES
    class_name_to_id = {n: i for i, n in enumerate(class_names)}
    y_train = tk.ml.ObjectsAnnotation.load_voc_0712_trainval(data_dir, class_name_to_id, without_difficult=True)
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(data_dir, class_name_to_id)
    return y_train, y_test, class_names


def load_data_csv(data_dir: pathlib.Path):
    """データの読み込み"""
    csv_dir = data_dir / 'csv'
    image_dir = csv_dir / 'images'
    class_names_path = csv_dir / 'class_names.txt'

    # クラス名の一覧
    class_names = list(filter(lambda x: len(x) > 0, tk.io.read_all_lines(class_names_path)))
    assert 'bg' not in class_names
    class_names = np.unique(['bg'] + class_names).tolist()
    class_name_to_id = {class_name: i for i, class_name in enumerate(class_names)}  # 'bg': 0

    y_train = _load_csv_annotations(csv_dir, 'train', image_dir, class_name_to_id)
    y_test = _load_csv_annotations(csv_dir, 'test', image_dir, class_name_to_id)

    return y_train, y_test, class_names


def _load_csv_annotations(csv_dir, data_type, image_dir, class_name_to_id):
    df = pd.read_csv(str(csv_dir / '{}.csv'.format(data_type)))
    # クラス名の一覧
    # ファイル名の一覧
    X = np.unique(df['filename'].values)
    y = []
    for x in tqdm(X, desc='images', ncols=100, ascii=True):
        # 画像サイズの取得
        with PIL.Image.open(image_dir / x) as img:
            width, height = img.size
        # class / bounding box
        df_target = df[df['filename'] == x]
        classes = [class_name_to_id[cn] for cn in df_target['class_name'].values]
        bboxes = [tpl for tpl in zip(df_target['left'].values / width, df_target['top'].values / height,
                                     df_target['right'].values / width, df_target['bottom'].values / height)]
        enables = [0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 for x1, y1, x2, y2 in bboxes]
        if not all(enables):
            warnings.warn('不正なデータ:\n{}'.format(df_target))
        if not any(enables):
            continue
        y.append(tk.ml.ObjectsAnnotation(
            folder='csv/images',  # data_dirからの相対パスにしておく
            filename=x,
            width=width,
            height=height,
            classes=[t for enable, t in zip(enables, classes) if enable],
            bboxes=[t for enable, t in zip(enables, bboxes) if enable],
            difficults=None))
    return np.array(y)


def load_data_pkl(data_dir: pathlib.Path):
    """データの読み込み"""
    y_train = joblib.load(data_dir / 'y_train.pkl')
    y_test = joblib.load(data_dir / 'y_test.pkl')
    class_names = joblib.load(data_dir / 'class_names.pkl')
    return y_train, y_test, class_names
