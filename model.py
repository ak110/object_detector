"""ObjectDetectionのモデル。"""
import warnings

import joblib
import numpy as np
import sklearn.cluster
import sklearn.metrics
from tqdm import tqdm

import model_net
import pytoolkit as tk

_TRAIN_DIFFICULT = False


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    # 取り出すfeature mapのサイズ
    FM_COUNTS = (40, 20, 10, 5)

    @classmethod
    def create(cls, nb_classes, y_train, pb_size_pattern_count=8):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。

        gridに配置したときのIOUを直接最適化するのは難しそうなので、
        とりあえず大雑把にKMeansでクラスタ化したりなど。
        """
        # 平均オブジェクト数
        mean_objets = np.mean([len(y.bboxes) for y in y_train])

        # bboxのサイズ
        bboxes = np.concatenate([y.bboxes for y in y_train])
        bboxes_size_list = bboxes[:, 2:] - bboxes[:, :2]
        bboxes_base_size_list = bboxes_size_list.min(axis=-1)  # 短辺のサイズのリスト
        # bboxes_base_size_list = np.sqrt(bboxes_size_list.prod(axis=-1))  # 縦幅・横幅の相乗平均のリスト

        # bboxのサイズごとにどこかのfeature mapに割り当てたことにして相対サイズをリストアップ
        tile_sizes = 1 / np.array(ObjectDetector.FM_COUNTS)
        pb_size_patterns = []
        for tile_size in tile_sizes:
            if tile_size == tile_sizes.max():
                mask = bboxes_base_size_list >= tile_size
            elif tile_size == tile_sizes.min():
                mask = bboxes_base_size_list < tile_size * 2
            else:
                mask = np.logical_and(bboxes_base_size_list >= tile_size, bboxes_base_size_list < tile_size * 2)
            pb_size_patterns.append(bboxes_size_list[mask] / tile_size)
        pb_size_patterns = np.concatenate(pb_size_patterns)
        assert len(pb_size_patterns.shape) == 2
        assert pb_size_patterns.shape[1] == 2

        # 相対サイズをクラスタリング
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_pattern_count, n_jobs=-1)
        cluster.fit(np.log(pb_size_patterns))
        pb_size_patterns = np.exp(cluster.cluster_centers_)
        assert pb_size_patterns.shape == (pb_size_pattern_count, 2)
        pb_size_patterns = pb_size_patterns[pb_size_patterns.prod(axis=-1).argsort(), :]  # 面積昇順に並べ替え (一応)

        return cls(nb_classes, mean_objets, pb_size_patterns)

    def __init__(self, nb_classes, mean_objets, pb_size_patterns):
        self.nb_classes = nb_classes
        self.input_size = model_net.get_input_size()
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。[1.5, 0.5] なら横が1.5倍、縦が0.5倍のものを用意。
        self.pb_size_patterns = pb_size_patterns.astype(np.float32)
        # 1 / fm_countに対する、prior boxの基準サイズの割合。(縦横の相乗平均)
        self.pb_size_ratios = np.sort(np.sqrt(pb_size_patterns[:, 0] * pb_size_patterns[:, 1]))
        # アスペクト比のリスト
        self.pb_aspect_ratios = np.sort(pb_size_patterns[:, 0] / pb_size_patterns[:, 1])

        # shape=(box数, 4)で座標
        self.pb_locs = None
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_info_indices = None
        # shape=(box数,4)でpred_locsのスケール
        self.pb_scales = None
        # 各prior boxの情報をdictで保持
        self.pb_info = None

        self._create_prior_boxes()

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4)
        assert self.pb_info_indices.shape == (nb_pboxes,)
        assert self.pb_scales.shape == (nb_pboxes, 4)
        assert len(self.pb_info) == len(ObjectDetector.FM_COUNTS) * len(self.pb_size_patterns)

    def _create_prior_boxes(self):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""
        self.pb_locs = []
        self.pb_info_indices = []
        self.pb_scales = []
        self.pb_info = []
        for fm_count in ObjectDetector.FM_COUNTS:
            # 敷き詰める間隔
            tile_size = 1.0 / fm_count
            # 敷き詰めたときの中央の位置のリスト
            lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, fm_count, dtype=np.float32)
            # 縦横に敷き詰め
            centers_x, centers_y = np.meshgrid(lin, lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes_base = np.concatenate((centers_x, centers_y), axis=1)
            # (x, y) → タイル×(x1, y1, x2, y2)
            prior_boxes_base = np.tile(prior_boxes_base, (1, 2))
            assert prior_boxes_base.shape == (fm_count ** 2, 4)

            for pb_pat in self.pb_size_patterns:
                # prior boxのサイズの基準
                pb_size = tile_size * pb_pat
                # prior boxのサイズ
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                prior_boxes = np.copy(prior_boxes_base)
                prior_boxes[:, 0] -= pb_size[0] / 2
                prior_boxes[:, 1] -= pb_size[1] / 2
                prior_boxes[:, 2] += pb_size[0] / 2
                prior_boxes[:, 3] += pb_size[1] / 2
                # はみ出ているのはclipしておく
                prior_boxes = np.clip(prior_boxes, 0, 1)

                self.pb_locs.extend(prior_boxes)
                self.pb_scales.extend([[pb_size[0], pb_size[1]] * 2] * len(prior_boxes))
                self.pb_info_indices.extend([len(self.pb_info)] * len(prior_boxes))
                self.pb_info.append({
                    'fm_count': fm_count,
                    'size': np.sqrt(pb_size.prod()),
                    'aspect_ratio': pb_size[0] / pb_size[1],
                    'count': len(prior_boxes),
                })

        self.pb_locs = np.array(self.pb_locs)
        self.pb_info_indices = np.array(self.pb_info_indices)
        self.pb_scales = np.array(self.pb_scales) * 0.1  # SSD風適当スケーリング

    def check_prior_boxes(self, logger, result_dir, y_test: [tk.ml.ObjectsAnnotation], class_names):
        """データに対して`self.pb_locs`がどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        match_counts = [0] * len(self.pb_info)  # iou >= 0.5のprior boxが1つ以上存在した回数
        unrec_widths = []  # iou < 0.5の横幅
        unrec_heights = []  # iou < 0.5の高さ
        unrec_ars = []  # iou < 0.5のアスペクト比
        rec_counts = []  # prior box毎の、iou >= 0.5の数
        rec_col_counts = []  # iou >= 0.5で2個以上のラベルが割り当たってしまったprior box数
        rec_delta_locs = []  # Δlocs
        for y in tqdm(y_test, desc='check_prior_boxes', ascii=True, ncols=100):
            # prior_boxesとbboxesで重なっているものを探す
            iou = tk.ml.compute_iou(self.pb_locs, y.bboxes)
            iou_mask = iou >= 0.5

            # クラスごとに再現率などを算出
            for gt_ix, class_id in enumerate(y.classes):
                assert 1 <= class_id < self.nb_classes
                m = iou_mask[:, gt_ix]
                success = m.any()
                y_true.append(class_id)
                y_pred.append(class_id if success else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                if success:
                    rec_counts.append(np.count_nonzero(m))
                    pb_ix = iou[:, gt_ix].argmax()
                    delta_locs = (y.bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]
                    match_counts[self.pb_info_indices[pb_ix]] += 1
                    rec_delta_locs.append(delta_locs)

            rec_col_counts.append(np.count_nonzero(np.count_nonzero(iou_mask, axis=1) >= 2))

            # 再現(iou >= 0.5)しなかったboxの情報を集める
            for bbox in y.bboxes[iou.max(axis=0) < 0.5]:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                unrec_widths.append(w)
                unrec_heights.append(h)
                unrec_ars.append(w / h)

        # 集計
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        total_gt_boxes = sum([len(y.bboxes) for y in y_test])
        cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)

        # ログ出力
        logger.debug(cr)
        logger.debug('[iou >= 0.5] counts:')
        for i, c in enumerate(match_counts):
            logger.debug('  prior boxes{fm=%d, size=%.2f ar=%.2f} = %d (%.02f%%)',
                         self.pb_info[i]['fm_count'],
                         self.pb_info[i]['size'],
                         self.pb_info[i]['aspect_ratio'],
                         c, 100 * c / self.pb_info[i]['count'] / total_gt_boxes)
        # Δlocの分布調査
        # mean≒0, std≒1とかくらいが学習しやすいはず。(SSDを真似た謎の0.1で大体そうなってる)
        delta_locs = np.concatenate(rec_delta_locs)
        logger.debug('delta loc: mean=%.2f std=%.2f min=%.2f max=%.2f',
                     delta_locs.mean(), delta_locs.std(), delta_locs.min(), delta_locs.max())
        # iou < 0.5の出現率
        logger.debug('[iou < 0.5] count: %d / %d (%.02f%%)',
                     len(unrec_widths), len(y_test), 100 * len(unrec_widths) / len(y_test))
        # iou >= 0.5での衝突の出現率
        logger.debug('[iou >= 0.5] collision count: %d / %d (%.02f%%)',
                     np.count_nonzero(rec_col_counts), len(y_test),
                     100 * np.count_nonzero(rec_col_counts) / len(y_test))

        # ヒストグラム色々を出力
        import matplotlib.pyplot as plt
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
        plt.hist([np.mean(np.abs(dl)) for dl in rec_delta_locs], bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('rec_mean_abs_delta.hist.png')))
        plt.close()
        plt.hist(rec_counts, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('rec_counts.hist.png')))
        plt.close()
        plt.hist(rec_col_counts, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('rec_col_counts.hist.png')))
        plt.close()

    def encode_truth(self, y_gt: [tk.ml.ObjectsAnnotation]):
        """学習用の`y_true`の作成。

        IOUが0.5以上のものを全部割り当てる＆割り当らなかったらIOUが一番大きいものに割り当てる。
        1つのprior boxに複数割り当たってしまっていたらIOUが一番大きいものを採用。
        """
        confs = np.zeros((len(y_gt), len(self.pb_locs), self.nb_classes), dtype=np.float32)
        confs[:, :, 0] = 1  # bg
        locs = np.zeros((len(y_gt), len(self.pb_locs), 4), dtype=np.float32)
        # 画像ごとのループ
        for i, y in enumerate(y_gt):
            bboxes = y.bboxes if _TRAIN_DIFFICULT or y.difficults is None else y.bboxes[np.logical_not(y.difficults)]

            # prior_boxesとbboxesで重なっているものを探す
            iou = tk.ml.compute_iou(self.pb_locs, bboxes)

            # 割り当てようとしているgt_ix
            pb_candidates = -np.ones((len(self.pb_locs),), dtype=int)  # -1埋め

            # IOUが0.5以上のものはまとめて割り当てる。1つのpboxに複数当たる場合はIOUが大きい方優先
            mask = iou.max(axis=1) >= 0.5
            pb_candidates[mask] = iou[mask, :].argmax(axis=1)

            # 0.5以上のものが無いgt_ixはIOUが一番大きいものに割り当てる
            failed_ix_list = np.where(iou.max(axis=0) < 0.5)[0]
            for gt_ix in failed_ix_list:
                pb_ix = iou[:, gt_ix].argmax()
                if False:  # ↓何故か毎回出てしまうためとりあえず無効化
                    if pb_candidates[pb_ix] >= 0:  # 割り当て済みな場合
                        if iou[pb_ix, pb_candidates[pb_ix]] < 0.5:
                            warnings.warn('IOU < 0.5での割り当ての重複: {}'.format(y.filename))
                pb_candidates[pb_ix] = gt_ix

            pb_ixs = np.where(pb_candidates >= 0)[0]
            assert len(pb_ixs) >= 1
            for pb_ix in pb_ixs:
                gt_ix = pb_candidates[pb_ix]
                class_id = y.classes[gt_ix]
                assert 0 < class_id < self.nb_classes
                # confs: 該当のクラスだけ>=_IOU_TH、残りはbg、他は0にする。
                confs[i, pb_ix, 0] = 0  # bg
                confs[i, pb_ix, class_id] = 1
                # locs: xmin, ymin, xmax, ymaxそれぞれのoffsetを回帰する。(スケーリングもする)
                locs[i, pb_ix, :] = (bboxes[gt_ix, :] - self.pb_locs[pb_ix, :]) / self.pb_scales[pb_ix, :]

        # いったんくっつける (損失関数の中で分割して使う)
        return np.concatenate([confs, locs], axis=-1)

    def select_predictions(self, classes_list, confs_list, locs_list, top_k=200, nms_threshold=0.45, parallel=None):
        """予測結果のうちスコアが高いものを取り出す。

        入出力は以下の3つの値。画像ごとにconfidenceの降順。
        - 画像数×BOX数×class_id
        - 画像数×BOX数×confidence
        - 画像数×BOX数×(xmin, ymin, xmax, ymax)

        BOX数は、入力はprior box数。出力は検出数(可変)
        """
        assert classes_list.shape[1:] == (len(self.pb_locs),)
        assert confs_list.shape[1:] == (len(self.pb_locs),)
        assert locs_list.shape[1:] == (len(self.pb_locs), 4)
        # 画像毎にループ
        if parallel:
            jobs = [joblib.delayed(self._select_prediction, check_pickle=False)(
                    pred_classes, pred_confs, pred_locs, top_k, nms_threshold)
                    for pred_classes, pred_confs, pred_locs
                    in zip(classes_list, confs_list, locs_list)]
            result_classes, result_confs, result_locs = zip(*parallel(jobs))
        else:
            result_classes = []
            result_confs = []
            result_locs = []
            for pred_classes, pred_confs, pred_locs in zip(classes_list, confs_list, locs_list):
                img_classes, img_confs, img_locs = self._select_prediction(
                    pred_classes, pred_confs, pred_locs, top_k, nms_threshold)
                result_classes.append(img_classes)
                result_confs.append(img_confs)
                result_locs.append(img_locs)
        assert len(result_classes) == len(classes_list)
        assert len(result_confs) == len(classes_list)
        assert len(result_locs) == len(classes_list)
        return result_classes, result_confs, result_locs

    def _select_prediction(self, pred_classes, pred_confs, pred_locs, top_k, nms_threshold):
        # 適当に上位のみ見る
        conf_mask = pred_confs.argsort()[::-1][:top_k * 4]
        pred_classes = pred_classes[conf_mask]
        pred_confs = pred_confs[conf_mask]
        pred_locs = pred_locs[conf_mask, :]
        # クラスごとに処理
        img_classes = []
        img_confs = []
        img_locs = []
        for target_class in range(1, self.nb_classes):
            targets = pred_classes == target_class
            if not targets.any():
                continue
            # 重複を除くtop_k個を採用
            target_confs = pred_confs[targets]
            target_locs = pred_locs[targets, :]
            idx = tk.ml.non_maximum_suppression(target_locs, target_confs, top_k, nms_threshold)
            # 結果
            img_classes.extend([target_class] * len(idx))
            img_confs.extend(target_confs[idx])
            img_locs.extend(target_locs[idx])
        # 最終結果 (confidence降順に並べつつもう一度top_k)
        img_confs = np.array(img_confs)
        sorted_indices = img_confs.argsort()[::-1][:top_k]
        img_confs = img_confs[sorted_indices]
        img_classes = np.array(img_classes)[sorted_indices]
        img_locs = np.array(img_locs)[sorted_indices]
        assert img_classes.shape == (len(sorted_indices),)
        assert img_confs.shape == (len(sorted_indices),)
        assert img_locs.shape == (len(sorted_indices), 4)
        return img_classes, img_confs, img_locs

    def loss(self, y_true, y_pred):
        """損失関数。"""
        import keras.backend as K
        import tensorflow as tf
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_confs, pred_locs, pred_iou = y_pred[:, :, :-5], y_pred[:, :, -5:-1], y_pred[:, :, -1]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)  # 各batch毎のobj数。
        # obj_countが1以上であることの確認
        with tf.control_dependencies([tf.assert_positive(obj_count)]):
            obj_count = tf.identity(obj_count)
        loss_conf = self._loss_conf(gt_confs, pred_confs, obj_count)
        loss_loc = self._loss_loc(gt_locs, pred_locs, obj_mask, obj_count)
        loss_iou = self._loss_iou(gt_locs, pred_locs, pred_iou, obj_mask, obj_count)
        return loss_conf + loss_loc + loss_iou

    @property
    def metrics(self):
        """各種metricをまとめて返す。"""
        return [self.loss_conf, self.loss_loc, self.loss_iou, self.acc_bg, self.acc_obj]

    @staticmethod
    def loss_conf(y_true, y_pred):
        """クラス分類の損失項。(metrics用)"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-5]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return ObjectDetector._loss_conf(gt_confs, pred_confs, obj_count)

    @staticmethod
    def loss_loc(y_true, y_pred):
        """位置の損失項。(metrics用)"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_locs = y_pred[:, :, -5:-1]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return ObjectDetector._loss_loc(gt_locs, pred_locs, obj_mask, obj_count)

    def loss_iou(self, y_true, y_pred):
        """IOUの損失項。(metrics用)"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
        pred_locs, pred_iou = y_pred[:, :, -5:-1], y_pred[:, :, -1]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return self._loss_iou(gt_locs, pred_locs, pred_iou, obj_mask, obj_count)

    @staticmethod
    def _loss_conf(gt_confs, pred_confs, obj_count):
        """分類のloss。"""
        import keras.backend as K
        if True:
            loss = tk.dl.categorical_focal_loss(gt_confs, pred_confs)
            loss = K.sum(loss, axis=-1) / obj_count  # normalized by the number of anchors assigned to a ground-truth box
        else:
            loss = tk.dl.categorical_crossentropy(gt_confs, pred_confs)
            loss = K.maximum(loss - 0.01, 0)  # clip
            loss = K.mean(loss, axis=-1)
        return loss

    @staticmethod
    def _loss_loc(gt_locs, pred_locs, obj_mask, obj_count):
        """位置のloss。"""
        import keras.backend as K
        loss = tk.dl.l1_smooth_loss(gt_locs, pred_locs)
        loss = K.sum(loss * obj_mask, axis=-1) / obj_count  # mean
        return loss

    def _loss_iou(self, gt_locs, pred_locs, pred_iou, obj_mask, obj_count):
        """IOUのloss。

        gt_locsとpred_locsからIOUを算出し、それとpred_iouのbinary crossentropyを返す。
        """
        import keras.backend as K
        bboxes_a = K.clip(gt_locs * self.pb_scales + self.pb_locs, 0, 1)
        bboxes_b = K.clip(pred_locs * self.pb_scales + self.pb_locs, 0, 1)
        lt = K.maximum(bboxes_a[:, :, :2], bboxes_b[:, :, :2])
        rb = K.minimum(bboxes_a[:, :, 2:], bboxes_b[:, :, 2:])
        area_inter = K.prod(rb - lt, axis=-1) * K.cast(K.all(K.less(lt, rb), axis=-1), K.floatx())
        area_a = K.maximum(K.prod(bboxes_a[:, :, 2:] - bboxes_a[:, :, :2], axis=-1), 0)
        area_b = K.maximum(K.prod(bboxes_b[:, :, 2:] - bboxes_b[:, :, :2], axis=-1), 0)
        area_union = area_a + area_b - area_inter
        iou = area_inter / area_union
        # IOUのcrossentropy
        loss = K.binary_crossentropy(iou, pred_iou)
        loss = K.sum(loss * obj_mask, axis=-1) / obj_count  # mean
        return loss

    @staticmethod
    def acc_bg(y_true, y_pred):
        """背景の再現率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-5]
        bg_mask = K.cast(K.greater_equal(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景
        bg_count = K.sum(bg_mask, axis=-1)
        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * bg_mask, axis=-1) / bg_count

    @staticmethod
    def acc_obj(y_true, y_pred):
        """物体の再現率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-4]
        pred_confs = y_pred[:, :, :-5]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * obj_mask, axis=-1) / obj_count

    def create_network(self):
        """ネットワークの作成"""
        return model_net.create_network(self)

    @staticmethod
    def create_pretrain_network(input_shape, nb_classes):
        """事前学習用ネットワークの作成"""
        return model_net.create_pretrain_network(input_shape, nb_classes)

    def create_predict_network(self, model):
        """予測用ネットワークの作成"""
        return model_net.create_predict_network(self, model)
