"""データのGenerator。"""
import numpy as np

import models
import pytoolkit as tk


def create_generator(image_size, preprocess_input, od: models.ObjectDetector):
    """ImageDataGeneratorを作って返す。"""
    def _transform(rgb: np.ndarray, y: tk.ml.ObjectsAnnotation, w, rand: np.random.RandomState, ctx: tk.generator.GeneratorContext):
        """変形を伴うAugmentation。"""
        assert ctx is not None

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
                    sb = np.round(bboxes * np.tile([image_size[1] / pw, image_size[0] / ph], 2))
                    if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいのはNG
                        continue
                    y.bboxes = bboxes / np.tile([pw, ph], 2)
                # 先に縮小
                rw = image_size[1] / pw
                rh = image_size[0] / ph
                new_w = int(round(old_w * rw))
                new_h = int(round(old_h * rh))
                rgb = tk.ndimage.resize(rgb, new_w, new_h, padding=None)
                # パディング
                px = int(round(px * rw))
                py = int(round(py * rh))
                padding = rand.choice(('edge', 'zero', 'one', 'rand'))
                rgb = tk.ndimage.pad_ltrb(rgb, px, py, image_size[1] - new_w - px, image_size[0] - new_h - py, padding, rand)
                assert rgb.shape[1] == image_size[1]
                assert rgb.shape[0] == image_size[0]
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
                    sb = np.round(bboxes * np.tile([image_size[1] / cw, image_size[0] / ch], 2))
                    if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいのはNG
                        continue
                    y.bboxes = bboxes / np.tile([cw, ch], 2)
                # 切り抜き
                rgb = tk.ndimage.crop(rgb, cx, cy, cw, ch)
                assert rgb.shape[1] == cw
                assert rgb.shape[0] == ch
                break

        return rgb, y, w

    def _process_output(y):
        if od is not None and y is not None:
            y = od.encode_truth([y])[0]
        return y

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.CustomAugmentation(_transform, probability=1))
    gen.add(tk.image.Resize(image_size))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.5),
        tk.image.RandomUnsharpMask(probability=0.5),
        tk.image.GaussianNoise(probability=0.5),
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ]))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(preprocess_input, batch_axis=True))
    gen.add(tk.image.ProcessOutput(_process_output))

    return gen


def _check():
    """お試しコード。"""
    import matplotlib
    matplotlib.use('Agg')

    from tqdm import tqdm

    import config

    save_dir = config.BASE_DIR / '___generator_check'
    save_dir.mkdir(exist_ok=True)

    class_names = ['bg'] + tk.ml.VOC_CLASS_NAMES
    class_name_to_id = {n: i for i, n in enumerate(class_names)}
    y_test = tk.ml.ObjectsAnnotation.load_voc_07_test(config.DATA_DIR, class_name_to_id)
    y_test = y_test[:1]
    X_test = tk.ml.ObjectsAnnotation.get_path_list(config.DATA_DIR, y_test)

    gen = create_generator((512, 512), od=None, preprocess_input=tk.image.preprocess_input_abs1)
    for i, (X_batch, y_batch) in zip(tqdm(range(16), ascii=True, ncols=100), gen.flow(X_test, y_test, data_augmentation=True)):
        for X, y in zip(X_batch, y_batch):
            X = tk.image.unpreprocess_input_abs1(X)
            tk.ml.plot_objects(
                X, save_dir / '{}.png'.format(i),
                y.classes, None, y.bboxes, class_names)


if __name__ == '__main__':
    _check()
