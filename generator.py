"""データのGenerator。"""
import numpy as np

import pytoolkit as tk
from model import ObjectDetector


class Generator(tk.image.ImageDataGenerator):
    """データのGenerator。"""

    def __init__(self, image_size, od: ObjectDetector):
        self.od = od
        super().__init__(image_size)
        self.add(0.125, tk.image.RandomBlur())
        self.add(0.125, tk.image.RandomBlur(partial=True))
        self.add(0.125, tk.image.RandomUnsharpMask())
        self.add(0.125, tk.image.RandomUnsharpMask(partial=True))
        self.add(0.125, tk.image.Sharp())
        self.add(0.125, tk.image.Sharp(partial=True))
        self.add(0.125, tk.image.Soft())
        self.add(0.125, tk.image.Soft(partial=True))
        self.add(0.125, tk.image.RandomMedian())
        self.add(0.125, tk.image.RandomMedian(partial=True))
        self.add(0.125, tk.image.GaussianNoise())
        self.add(0.125, tk.image.GaussianNoise(partial=True))
        self.add(0.125, tk.image.RandomSaturation())
        self.add(0.125, tk.image.RandomBrightness())
        self.add(0.125, tk.image.RandomContrast())
        self.add(0.125, tk.image.RandomLighting())

    def _prepare(self, X, y=None, weights=None, parallel=None, data_augmentation=False, rand=None):
        """画像の読み込みとDataAugmentation。"""
        X, y, weights = super()._prepare(X, y, weights, parallel, data_augmentation, rand)
        if self.od is not None and y[0] is not None:
            # 学習用に変換
            y = self.od.encode_truth(y)
        return X, y, weights

    def _transform(self, rgb: np.ndarray, y: tk.ml.ObjectsAnnotation, w, rand: np.random.RandomState):
        """変形を伴うAugmentation。"""
        # 左右反転
        if rand.rand() <= 0.5:
            rgb = tk.ndimage.flip_lr(rgb)
            if y is not None:
                y.bboxes[:, [0, 2]] = 1 - y.bboxes[:, [2, 0]]
        # padding
        if rand.rand() <= 0.5:
            old_w, old_h = rgb.shape[1], rgb.shape[0]
            pad_x1 = int(round(old_w * rand.uniform(0, 0.5)))
            pad_y1 = int(round(old_h * rand.uniform(0, 0.5)))
            pad_x2 = int(round(old_w * rand.uniform(0, 0.5)))
            pad_y2 = int(round(old_h * rand.uniform(0, 0.5)))
            rgb = tk.ndimage.pad_ltrb(rgb, pad_x1, pad_y1, pad_x2, pad_y2)
            if y is not None:
                assert rgb.shape[1] == pad_x1 + old_w + pad_x2
                assert rgb.shape[0] == pad_y1 + old_h + pad_y2
                y.bboxes[:, 0] = (pad_x1 / rgb.shape[1]) + y.bboxes[:, 0] * (old_w / rgb.shape[1])
                y.bboxes[:, 1] = (pad_y1 / rgb.shape[0]) + y.bboxes[:, 1] * (old_h / rgb.shape[0])
                y.bboxes[:, 2] = (pad_x1 / rgb.shape[1]) + y.bboxes[:, 2] * (old_w / rgb.shape[1])
                y.bboxes[:, 3] = (pad_y1 / rgb.shape[0]) + y.bboxes[:, 3] * (old_h / rgb.shape[0])
        return rgb, y, w


def _check():
    """お試しコード。"""
    import better_exceptions
    better_exceptions.MAX_LENGTH = 128

    import pathlib
    from tqdm import tqdm
    from voc_data import CLASS_NAMES, load_data
    base_dir = pathlib.Path(__file__).parent
    data_dir = base_dir.joinpath('data')
    save_dir = base_dir.joinpath('___generator_check')
    save_dir.mkdir(exist_ok=True)

    (_, _), (X_test, y_test) = load_data(data_dir, True, 1)

    gen = Generator((512, 512), od=None)
    for i, (X_batch, y_batch) in zip(tqdm(range(16), ascii=True, ncols=128), gen.flow(X_test, y_test, data_augmentation=True)):
        for X, y in zip(X_batch, y_batch):
            X = tk.image.unpreprocess_input_abs1(X)
            tk.ml.plot_objects(
                X, save_dir.joinpath('{}.png'.format(i)),
                y.classes, None, y.bboxes, CLASS_NAMES)


if __name__ == '__main__':
    _check()
