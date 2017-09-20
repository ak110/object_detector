"""データのGenerator。"""

import pytoolkit as tk
from model import ObjectDetector


class Generator(tk.image.ImageDataGenerator):
    """データのGenerator。"""

    def __init__(self, image_size, od: ObjectDetector):
        super().__init__(image_size)
        self.od = od
        # self.add(0.5, tk.image.FlipLR())  # TODO: _transformでyと合わせてやる必要あり
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
        if y is not None:
            # 学習用に変換
            y = self.od.encode_truth(y)
        return X, y, weights

    def _transform(self, rgb, rand):
        """変形を伴うAugmentation。"""
        # とりあえず殺しておく
        return rgb
