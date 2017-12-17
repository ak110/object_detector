"""データのGenerator。"""
import numpy as np

import model_net
import models
import pytoolkit as tk


class Generator(tk.image.ImageDataGenerator):
    """データのGenerator。"""

    def __init__(self, image_size, od: models.ObjectDetector, base_network: str):
        self.od = od
        super().__init__(image_size, preprocess_input=model_net.get_preprocess_input(base_network))
        self.add(0.5, tk.image.RandomErasing())
        self.add(1.0, tk.image.RandomErasing(object_aware=True, object_aware_prob=0.5))
        self.add(0.25, tk.image.RandomBlur())
        self.add(0.25, tk.image.RandomBlur(partial=True))
        self.add(0.25, tk.image.RandomUnsharpMask())
        self.add(0.25, tk.image.RandomUnsharpMask(partial=True))
        self.add(0.25, tk.image.RandomMedian())
        self.add(0.25, tk.image.GaussianNoise())
        self.add(0.25, tk.image.GaussianNoise(partial=True))
        self.add(0.5, tk.image.RandomSaturation())
        self.add(0.5, tk.image.RandomBrightness())
        self.add(0.5, tk.image.RandomContrast())
        self.add(0.5, tk.image.RandomHue())

    def _load(self, x, y, w, data_augmentation, seed):
        rgb, y, w = super()._load(x, y, w, data_augmentation, seed)

        if self.od is not None and y is not None:
            y = self.od.encode_truth([y])[0]

        return rgb, y, w

    def _transform(self, rgb: np.ndarray, y: tk.ml.ObjectsAnnotation, w, rand: np.random.RandomState):
        """変形を伴うAugmentation。"""
        # 左右反転
        if rand.rand() <= 0.5:
            rgb = tk.ndimage.flip_lr(rgb)
            if y is not None:
                y.bboxes[:, [0, 2]] = 1 - y.bboxes[:, [2, 0]]

        aspect_rations = (3 / 4, 4 / 3)
        aspect_prob = 0.5
        ar = np.sqrt(rand.choice(aspect_rations)) if rand.rand() <= aspect_prob else 1
        # padding or crop
        if rand.rand() <= 0.5:
            # padding (zoom out)
            old_w, old_h = rgb.shape[1], rgb.shape[0]
            for _ in range(30):
                pr = np.random.uniform(1.5, 4)  # SSDは16倍とか言っているが、やり過ぎな気がするので適当
                pw = max(old_w, int(round(old_w * pr * ar)))
                ph = max(old_h, int(round(old_h * pr / ar)))
                px = rand.randint(0, pw - old_w + 1)
                py = rand.randint(0, ph - old_h + 1)
                if y is not None:
                    bboxes = np.copy(y.bboxes)
                    bboxes[:, (0, 2)] = px + bboxes[:, (0, 2)] * old_w
                    bboxes[:, (1, 3)] = py + bboxes[:, (1, 3)] * old_h
                    sb = np.round(bboxes * np.tile([self.image_size[1] / pw, self.image_size[0] / ph], 2))
                    if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいのはNG
                        continue
                    y.bboxes = bboxes / np.tile([pw, ph], 2)
                # 先に縮小
                rw = self.image_size[1] / pw
                rh = self.image_size[0] / ph
                new_w = int(round(old_w * rw))
                new_h = int(round(old_h * rh))
                interp = rand.choice(['nearest', 'lanczos', 'bilinear', 'bicubic'])
                rgb = tk.ndimage.resize(rgb, new_w, new_h, padding=None, interp=interp)
                # パディング
                px = int(round(px * rw))
                py = int(round(py * rh))
                padding = rand.choice(('same', 'zero', 'rand'))
                rgb = tk.ndimage.pad_ltrb(rgb, px, py, self.image_size[1] - new_w - px, self.image_size[0] - new_h - py, padding, rand)
                assert rgb.shape[1] == self.image_size[1]
                assert rgb.shape[0] == self.image_size[0]
                break
        else:
            # crop (zoom in)
            # SSDでは結構複雑なことをやっているが、とりあえず簡単に実装
            bb_center = (y.bboxes[:, 2:] - y.bboxes[:, :2]) / 2  # y.bboxesの中央の座標
            bb_center *= [rgb.shape[1], rgb.shape[0]]
            for _ in range(30):
                crop_rate = 0.15625
                cr = rand.uniform(1 - crop_rate, 1)  # 元のサイズに対する割合
                cw = min(rgb.shape[1], int(np.floor(rgb.shape[1] * cr * ar)))
                ch = min(rgb.shape[0], int(np.floor(rgb.shape[0] * cr / ar)))
                cx = rand.randint(0, rgb.shape[1] - cw + 1)
                cy = rand.randint(0, rgb.shape[0] - ch + 1)
                # 全bboxの中央の座標を含む場合のみOKとする
                if np.logical_or(bb_center < [cx, cy], [cx + cw, cy + ch] < bb_center).any():
                    continue
                if y is not None:
                    bboxes = np.copy(y.bboxes)
                    bboxes[:, (0, 2)] = np.round(bboxes[:, (0, 2)] * rgb.shape[1] - cx)
                    bboxes[:, (1, 3)] = np.round(bboxes[:, (1, 3)] * rgb.shape[0] - cy)
                    bboxes = np.clip(bboxes, 0, 1)
                    sb = np.round(bboxes * np.tile([self.image_size[1] / cw, self.image_size[0] / ch], 2))
                    if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいのはNG
                        continue
                    y.bboxes = bboxes / np.tile([cw, ch], 2)
                # 切り抜き
                rgb = tk.ndimage.crop(rgb, cx, cy, cw, ch)
                assert rgb.shape[1] == cw
                assert rgb.shape[0] == ch
                break

        return rgb, y, w


def _check():
    """お試しコード。"""
    import better_exceptions
    better_exceptions.MAX_LENGTH = 128
    import matplotlib
    matplotlib.use('Agg')

    import data
    import pathlib
    from tqdm import tqdm
    base_dir = pathlib.Path(__file__).parent
    data_dir = base_dir.joinpath('data')
    save_dir = base_dir.joinpath('___generator_check')
    save_dir.mkdir(exist_ok=True)

    (_, _), (X_test, y_test), class_names = data.load_data(data_dir, 'voc')
    X_test = X_test[:1]
    y_test = y_test[:1]

    gen = Generator((512, 512), od=None, base_network='custom')
    for i, (X_batch, y_batch) in zip(tqdm(range(16), ascii=True, ncols=128), gen.flow(X_test, y_test, data_augmentation=True)):
        for X, y in zip(X_batch, y_batch):
            X = tk.image.unpreprocess_input_abs1(X)
            tk.ml.plot_objects(
                X, save_dir.joinpath('{}.png'.format(i)),
                y.classes, None, y.bboxes, class_names)


if __name__ == '__main__':
    _check()
