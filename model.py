"""ObjectDetectionのモデル。"""
import warnings

import numpy as np
import sklearn.cluster
import sklearn.metrics
from tqdm import tqdm

import pytoolkit as tk

_LARGE_IMAGE_SIZE = False
_TRAIN_DIFFICULT = True


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    # 取り出すfeature mapのサイズ
    FM_COUNTS = (40, 20, 10, 5)

    @classmethod
    def create(cls, nb_classes, y_train, pb_size_patterns=3, aspect_ratio_patterns=5):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。

        gridに配置したときのIOUを直接最適化するのは難しそうなので、
        とりあえず大雑把にKMeansでクラスタ化したりなど。
        """
        bboxes = np.concatenate([y.bboxes for y in y_train])
        # 平均オブジェクト数
        mean_objets = np.mean([len(y.bboxes) for y in y_train])
        # サイズ(feature mapのサイズからの相対値)
        sizes = np.sqrt((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
        fm_sizes = 1 / np.array(ObjectDetector.FM_COUNTS)
        pb_size_ratios = np.concatenate([sizes / fm_size for fm_size in fm_sizes])
        pb_size_ratios = pb_size_ratios[np.logical_and(pb_size_ratios >= 1, pb_size_ratios < 5)]
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_patterns, n_jobs=-1)
        cluster.fit(np.expand_dims(pb_size_ratios, axis=-1))
        pb_size_ratios = np.sort(cluster.cluster_centers_[:, 0])
        # アスペクト比
        log_ars = np.log((bboxes[:, 2] - bboxes[:, 0]) / (bboxes[:, 3] - bboxes[:, 1]))
        cluster = sklearn.cluster.KMeans(n_clusters=aspect_ratio_patterns, n_jobs=-1)
        cluster.fit(np.expand_dims(log_ars, axis=-1))
        aspect_ratios = np.sort(np.exp(cluster.cluster_centers_[:, 0]))

        return cls(nb_classes, mean_objets, pb_size_ratios, aspect_ratios)

    def __init__(self, nb_classes, mean_objets, pb_size_ratios=(1.5, 2.5), aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3)):
        self.nb_classes = nb_classes
        self.input_size = (645, 645) if _LARGE_IMAGE_SIZE else (320, 320)
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。3なら3倍の大きさのものを用意。
        self.pb_size_ratios = np.array(pb_size_ratios)
        # アスペクト比のリスト
        self.aspect_ratios = np.array(aspect_ratios)

        # shape=(box数, 4)で座標
        self.pb_locs = None
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_indices = None
        # shape=(box数,4)でpred_locsのスケール
        self.pb_scales = None
        # shape=(config数,)でpb_size
        self.pb_sizes = None
        # 各prior boxの情報をdictで保持
        self.config = None

        self._create_prior_boxes()

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4)
        assert self.pb_indices.shape == (nb_pboxes,)
        assert self.pb_scales.shape == (nb_pboxes, 4)
        assert self.pb_sizes.shape == (len(ObjectDetector.FM_COUNTS) * len(self.pb_size_ratios),)
        assert len(self.config) == len(ObjectDetector.FM_COUNTS) * len(self.pb_size_ratios)

    def _create_prior_boxes(self):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""
        self.pb_locs = []
        self.pb_indices = []
        self.pb_scales = []
        self.pb_sizes = []
        self.config = []
        for fm_count in ObjectDetector.FM_COUNTS:
            # 敷き詰める間隔
            tile_size = 1.0 / fm_count
            # 敷き詰めたときの中央の位置のリスト
            lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, fm_count, dtype=np.float32)
            # 縦横に敷き詰め
            centers_x, centers_y = np.meshgrid(lin, lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
            # (x, y) → サイズ比アスペクト比×タイル×(x1, y1, x2, y2)
            prior_boxes = np.tile(prior_boxes, (len(self.pb_size_ratios) * len(self.aspect_ratios), 1, 2))
            assert prior_boxes.shape == (len(self.pb_size_ratios) * len(self.aspect_ratios), fm_count ** 2, 4)

            # #  → タイル×サイズ比アスペクト比×(x1, y1, x2, y2)
            # prior_boxes = np.swapaxes(prior_boxes, 0, 1)
            # assert prior_boxes.shape == (fm_count ** 2, len(self.pb_size_ratios) * len(self.aspect_ratios), 4)
            # half_box_widths = np.reshape([0.5 * tile_size * psr * np.sqrt(self.aspect_ratios) for psr in self.pb_size_ratios], (1, -1))
            # half_box_heights = np.reshape([0.5 * tile_size * psr / np.sqrt(self.aspect_ratios) for psr in self.pb_size_ratios], (1, -1))

            # prior boxのサイズ
            # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
            half_box_widths = np.reshape([0.5 * tile_size * psr * np.sqrt(self.aspect_ratios) for psr in self.pb_size_ratios], (-1, 1))
            half_box_heights = np.reshape([0.5 * tile_size * psr / np.sqrt(self.aspect_ratios) for psr in self.pb_size_ratios], (-1, 1))
            prior_boxes[:, :, 0] -= half_box_widths
            prior_boxes[:, :, 1] -= half_box_heights
            prior_boxes[:, :, 2] += half_box_widths
            prior_boxes[:, :, 3] += half_box_heights
            # はみ出ているのはclipしておく
            prior_boxes = np.clip(prior_boxes, 0, 1)

            # (タイル, サイズ比アスペクト比, 4) → (タイルサイズ比アスペクト比, 4)
            prior_boxes = prior_boxes.reshape(-1, 4)

            self.pb_locs.extend(prior_boxes)
            self.pb_indices.extend([len(self.config)] * len(prior_boxes))
            self.pb_scales.extend(np.tile(prior_boxes[:, 2:] - prior_boxes[:, :2], 2))
            for pb_size in tile_size * self.pb_size_ratios:
                self.pb_sizes.append(pb_size)
                self.config.append({'fm_count': fm_count, 'pb_size': pb_size, 'box_count': len(prior_boxes)})

        self.pb_locs = np.array(self.pb_locs)
        self.pb_indices = np.array(self.pb_indices)
        self.pb_sizes = np.array(self.pb_sizes)
        self.pb_scales = np.array(self.pb_scales)

    def check_prior_boxes(self, logger, result_dir, y_test: [tk.ml.ObjectsAnnotation], class_names):
        """データに対して`self.pb_locs`がどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        match_counts = [0 for _ in range(len(self.config))]
        rec_mean_abs_delta = []
        unrec_widths = []
        unrec_heights = []
        unrec_ars = []
        for y in tqdm(y_test, desc='check_prior_boxes', ascii=True, ncols=100):
            iou = tk.ml.compute_iou(self.pb_locs, y.bboxes)
            # クラスごとに再現率を求める
            for gt_ix, class_id in enumerate(y.classes):
                assert 1 <= class_id < self.nb_classes
                m = iou[:, gt_ix] >= 0.5
                success = m.any()
                y_true.append(class_id)
                y_pred.append(class_id if success else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                if success:
                    pb_ix = self.pb_indices[iou[:, gt_ix].argmax()]
                    mean_abs_delta = np.mean(np.abs((y.bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]))
                    match_counts[pb_ix] += 1
                    rec_mean_abs_delta.append(mean_abs_delta)
            # 再現(iou >= 0.5)しなかったboxの情報を集める
            for bbox in y.bboxes[iou.max(axis=0) < 0.5]:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                unrec_widths.append(w)
                unrec_heights.append(h)
                unrec_ars.append(w / h)
        # 再現率などの表示
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        total_gt_boxes = sum([len(y.bboxes) for y in y_test])
        cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)
        logger.debug('prior boxs = %d', len(self.pb_locs))
        logger.debug(cr)
        logger.debug('match counts:')
        for i, c in enumerate(match_counts):
            logger.debug('  prior_boxes_%d = %d (%.02f%%)',
                         self.config[i]['fm_count'], c,
                         100 * c / self.config[i]['box_count'] / total_gt_boxes)
        # 再現しなかったboxのwidth/height/aspect ratioのヒストグラムを出力
        import matplotlib.pyplot as plt
        plt.hist(rec_mean_abs_delta, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('rec_mean_abs_delta.hist.png')))
        plt.close()
        plt.hist(unrec_widths, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_widths.hist.png')))
        plt.close()
        plt.hist(unrec_heights, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_heights.hist.png')))
        plt.close()
        plt.xscale('log')
        plt.hist(unrec_ars, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('unrec_ars.hist.png')))
        plt.close()

    @staticmethod
    def loss(y_true, y_pred):
        """損失関数。"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_confs, pred_locs = y_pred[:, :, :-4], y_pred[:, :, -4:]

        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)  # 各batch毎のobj数。
        obj_div = K.maximum(obj_count, K.ones_like(obj_count))

        loss_conf = ObjectDetector._loss_conf(gt_confs, pred_confs, obj_div)
        loss_loc = ObjectDetector._loss_loc(gt_locs, pred_locs, obj_mask, obj_div)
        return loss_conf + loss_loc

    @staticmethod
    def _loss_conf(gt_confs, pred_confs, obj_div):
        """分類のloss。

        # alpha

        Focal lossの論文では0.25が良いとしているが、他のパラメータとのバランスに依ると思う。

        SSDではbg : obj = 3 : 1になるようにminingして学習している。
        bg : obj = (priorbox - assigned) * (1 - alpha) : assigned * alpha
        なので、
        alpha = (priorbox - assigned) / (priorbox + assigned * (3 - 1))
        となる。
        priorbox = 46035
        assigned = 60
        とすると、
        alpha = (46035 - 60) / (46035 + 60 * (3 - 1)) = 0.996

        ただし、miningはlossの大きいもの優先なので必ずしもこの通りではない。
        仮に3→9とすると0.988。

        """
        import keras.backend as K
        loss = tk.dl.categorical_focal_loss(gt_confs, pred_confs, alpha=0.99)
        loss = K.sum(loss, axis=-1) / obj_div  # normalized by the number of anchors assigned to a ground-truth box
        return loss

    @staticmethod
    def _loss_loc(gt_locs, pred_locs, obj_mask, obj_div):
        """位置のloss。"""
        import keras.backend as K
        loss = tk.dl.l1_smooth_loss(gt_locs, pred_locs)
        loss = K.sum(loss * obj_mask, axis=-1) / obj_div  # mean
        return loss

    @staticmethod
    def loss_loc(y_true, y_pred):
        """位置の損失項。"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_locs = y_pred[:, :, -4:]

        obj_mask = gt_confs[:, :, 0] < 0.5  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)
        obj_div = K.maximum(obj_count, K.ones_like(obj_count))

        return ObjectDetector._loss_loc(gt_locs, pred_locs, obj_mask, obj_div)

    @staticmethod
    def acc_bg(y_true, y_pred):
        """背景の正解率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-4]

        bg_mask = gt_confs[:, :, 0] >= 0.5  # 背景
        bg_mask = K.cast(bg_mask, K.floatx())
        bg_count = K.sum(bg_mask, axis=-1)

        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * bg_mask, axis=-1) / bg_count

    @staticmethod
    def acc_obj(y_true, y_pred):
        """物体の正解率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-4]

        obj_mask = gt_confs[:, :, 0] < 0.5  # 背景以外
        obj_mask = K.cast(obj_mask, K.floatx())
        obj_count = K.sum(obj_mask, axis=-1)

        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * obj_mask, axis=-1) / obj_count

    def encode_truth(self, y_gt: [tk.ml.ObjectsAnnotation]):
        """学習用の`y_true`の作成。

        IOUが0.5以上のものを全部割り当てる＆割り当らなかったらIOUが一番大きいものに割り当てる。
        1つのprior boxに複数割り当たってしまっていたらIOUが一番大きいものを採用。
        """
        confs = np.zeros((len(y_gt), len(self.pb_locs), self.nb_classes), dtype=np.float32)
        confs[:, :, 0] = 1
        locs = np.zeros((len(y_gt), len(self.pb_locs), 4), dtype=np.float32)
        # 画像ごとのループ
        for i, y in enumerate(y_gt):
            # prior_boxesとbboxesで重なっているものを探す
            iou = tk.ml.compute_iou(self.pb_locs, y.bboxes)

            pb_confs = np.zeros((len(self.pb_locs),), dtype=float)  # 割り当てようとしているconfidence
            pb_candidates = -np.ones((len(self.pb_locs),), dtype=int)  # 割り当てようとしているgt_ix

            # 大きいものから順に割り当てる (小さいものが出来るだけうまく割当たって欲しいので)
            gt_ix_list = np.argsort(-np.prod(y.bboxes[:, 2:] - y.bboxes[:, :2], axis=-1))
            for gt_ix in gt_ix_list:
                if not _TRAIN_DIFFICULT and y.difficult[gt_ix]:
                    continue
                pb_ixs = np.where(iou[:, gt_ix] >= 0.5)[0]
                if pb_ixs.any():
                    # IOUが0.5以上のものがあるなら全部割り当てる
                    iou_list = iou[pb_ixs, gt_ix]
                    obj_confs = iou_list / iou_list.max()  # iouが最大のもの以外はconfが低めになるように割合で割り当てる
                    assert (obj_confs > 0.5).all()
                    pb_confs[pb_ixs] = obj_confs
                    pb_candidates[pb_ixs] = gt_ix
                else:
                    # 0.5以上のものが無ければIOUが一番大きいものに割り当てる
                    pb_ixs = iou[:, gt_ix].argmax()
                    if pb_candidates[pb_ixs] >= 0:  # 割り当て済みな場合
                        if iou[pb_ixs, pb_candidates[pb_ixs]] < 0.5:
                            warnings.warn(
                                'IOU < 0.5での割り当ての重複: {} ({}, {})'.format(
                                    y.filename, pb_candidates[pb_ixs], gt_ix))
                    pb_confs[pb_ixs] = 1
                    pb_candidates[pb_ixs] = gt_ix

            for pb_ix in np.where(pb_candidates >= 0)[0]:
                conf = pb_confs[pb_ix]
                gt_ix = pb_candidates[pb_ix]
                class_id = y.classes[gt_ix]
                assert 0 < class_id < self.nb_classes
                assert 0 < conf <= 1
                # confs: 該当のクラスだけ1にする。
                confs[i, pb_ix, 0] = 1 - conf  # bg
                confs[i, pb_ix, class_id] = conf
                # locs: xmin, ymin, xmax, ymaxそれぞれのoffsetをそのまま回帰する。(prior_boxのサイズでスケーリングもする)
                locs[i, pb_ix, :] = (y.bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]

        # 分類の教師が合計≒1になっていることの確認
        assert (np.abs(confs.sum(axis=-1) - 1) < 1e-7).all()

        # いったんくっつける (損失関数の中で分割して使う)
        return np.concatenate([confs, locs], axis=-1)

    def decode_predictions(self, predictions, top_k=16, detect_least_conf=0.9, detect_min_conf=0.6, collision_iou=0.75):
        """予測結果をデコードする。

        出力は以下の3つの値。画像ごとにconfidenceの降順。
        - class_id×検出数×画像数
        - confidence×検出数×画像数
        - (xmin, ymin, xmax, ymax)×検出数×画像数

        """
        assert predictions.shape[1] == len(self.pb_locs)
        assert predictions.shape[2] == self.nb_classes + 4
        confs_list, locs_list = predictions[:, :, :-4], predictions[:, :, -4:]
        classes, confs, locs = zip(*[
            self._decode_prediction(confs, locs, top_k, detect_least_conf, detect_min_conf, collision_iou)
            for confs, locs
            in zip(confs_list, locs_list)])
        return classes, confs, locs

    def _decode_prediction(self, pred_confs, pred_locs, top_k, detect_least_conf, detect_min_conf, collision_iou):
        """予測結果のデコード。(1枚分)"""
        assert len(pred_confs) == len(pred_locs)
        assert pred_confs.shape == (len(self.pb_locs), self.nb_classes)
        assert pred_locs.shape == (len(self.pb_locs), 4)
        classes = []
        confs = []
        locs = []
        detections = 0
        # confidenceの上位を降順にループ
        obj_confs = pred_confs[:, 1:].max(axis=-1)  # prior box数分の、背景以外のconfidenceの最大値
        for pb_ix in np.argsort(obj_confs)[::-1]:
            loc = self.pb_locs[pb_ix, :] + pred_locs[pb_ix, :] * self.pb_scales[pb_ix, :]
            loc = np.clip(loc, 0, 1)  # はみ出ている分はクリッピング
            assert loc.shape == (4,)
            # 幅や高さが0以下になってしまっているものはスキップする
            if (loc[2:] <= loc[:2]).any():
                continue
            # 充分高いconfidenceのものがあるなら、ある程度以下のconfidenceのものは無視
            if len(confs) >= 1 and detect_least_conf <= confs[0] and obj_confs[pb_ix] < detect_min_conf:
                break

            detections += 1
            # 最大top_k個まで検知扱いにする
            if top_k < detections:
                break

            # 既に出現済みのものと大きく重なっているものはスキップする
            class_id = pred_confs[pb_ix, 1:].argmax(axis=-1) + 1
            if len(locs) >= 1:
                col_iou = tk.ml.compute_iou(np.array(locs), np.array([loc]))
                assert col_iou.shape == (len(locs), 1)
                col_mask = col_iou[:, 0] >= collision_iou
                if any([class_id == classes[col_ix] for col_ix in np.where(col_mask)[0]]):
                    continue

            classes.append(class_id)
            confs.append(obj_confs[pb_ix])
            locs.append(loc)

        return np.array(classes), np.array(confs), np.array(locs)
