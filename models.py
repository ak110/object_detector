"""ObjectDetectionのモデル。"""
import pathlib

import numpy as np
import sklearn.cluster
import sklearn.externals.joblib as joblib
import sklearn.metrics

import pytoolkit as tk

_VAR_LOC = 0.2  # SSD風(?)適当スケーリング
_VAR_SIZE = 0.2  # SSD風適当スケーリング


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    @classmethod
    @tk.log.trace()
    def create(cls, base_network, input_size, map_sizes, nb_classes, y_train, pb_size_pattern_count=8):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。

        gridに配置したときのIOUを直接最適化するのは難しそうなので、
        とりあえず大雑把にKMeansでクラスタ化したりなど。
        """
        logger = tk.log.get(__name__)
        logger.info('ハイパーパラメータ算出: base-network=%s input-size=%s map-sizes=%s classes=%d',
                    base_network, input_size, map_sizes, nb_classes)

        map_sizes = np.array(sorted(map_sizes)[::-1])
        mean_objets = np.mean([len(y.bboxes) for y in y_train])

        # bboxのサイズ
        bboxes = np.concatenate([y.bboxes for y in y_train])
        bboxes_sizes = bboxes[:, 2:] - bboxes[:, :2]
        bb_base_sizes = bboxes_sizes.min(axis=-1)  # 短辺のサイズのリスト
        assert (bb_base_sizes >= 0).all()  # 不正なデータは含まれない想定 (手抜き)
        assert (bb_base_sizes < 1).all()  # 不正なデータは含まれない想定 (手抜き)

        # bboxのサイズごとにどこかのfeature mapに割り当てたことにして相対サイズをリストアップ
        tile_sizes = 1 / map_sizes
        min_area_pattern = 0.5  # グリッドのサイズの半分を下限としてみる
        max_area_pattern = 1 / tile_sizes.max()  # 一番荒いmapが画像全体になるくらいのスケールを上限としてみる
        bboxes_size_patterns = []
        for tile_size in tile_sizes:
            size_pattern = bboxes_sizes / tile_size
            area_pattern = np.sqrt(size_pattern.prod(axis=-1))  # 縦幅・横幅の相乗平均のリスト
            mask = np.logical_and(min_area_pattern <= area_pattern, area_pattern <= max_area_pattern)
            bboxes_size_patterns.append(size_pattern[mask])
        bboxes_size_patterns = np.concatenate(bboxes_size_patterns)
        assert len(bboxes_size_patterns.shape) == 2
        assert bboxes_size_patterns.shape[1] == 2

        # リストアップしたものをクラスタリング。(YOLOv2のDimension Clustersのようなもの)。
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_pattern_count, n_jobs=-1)
        pb_size_patterns = cluster.fit(bboxes_size_patterns).cluster_centers_
        assert pb_size_patterns.shape == (pb_size_pattern_count, 2)

        return cls(base_network, input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns)

    def __init__(self, base_network, input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns):
        self.base_network = base_network
        self.image_size = (input_size, input_size)
        self.map_sizes = np.array(map_sizes)
        self.nb_classes = nb_classes
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。[1.5, 0.5] なら横が1.5倍、縦が0.5倍のものを用意。
        # 面積昇順に並べ替えておく。
        self.pb_size_patterns = pb_size_patterns[pb_size_patterns.prod(axis=-1).argsort(), :].astype(np.float32)
        # 1 / fm_countに対する、prior boxの基準サイズの割合。(縦横の相乗平均)
        self.pb_size_ratios = np.sort(np.sqrt(pb_size_patterns[:, 0] * pb_size_patterns[:, 1]))
        # アスペクト比のリスト
        self.pb_aspect_ratios = np.sort(pb_size_patterns[:, 0] / pb_size_patterns[:, 1])

        # shape=(box数, 4)で座標(アスペクト比などを適用する前のグリッド)
        self.pb_grid = []
        # shape=(box数, 4)で座標
        self.pb_locs = []
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_info_indices = []
        # shape=(box数,2)でprior boxの中心の座標
        self.pb_centers = []
        # shape=(box数,2)でprior boxのサイズ
        self.pb_sizes = []
        # 各prior boxの情報をdictで保持
        self.pb_info = []

        self._add_prior_boxes(self.map_sizes, self.pb_size_patterns)
        self.pb_grid = np.array(self.pb_grid)
        self.pb_locs = np.array(self.pb_locs)
        self.pb_info_indices = np.array(self.pb_info_indices)
        self.pb_centers = np.array(self.pb_centers)
        self.pb_sizes = np.array(self.pb_sizes)

        # prior boxの重心はpb_gridの中心であるはず
        ct = np.mean([self.pb_locs[:, 2:], self.pb_locs[:, :2]], axis=0)
        assert np.logical_and(self.pb_grid[:, :2] <= ct, ct < self.pb_grid[:, 2:]).all()
        # はみ出ているprior boxは使用しないようにする
        self.pb_mask = np.logical_and(self.pb_locs >= -0.1, self.pb_locs <= +1.1).all(axis=-1)

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4), f'shape error: {self.pb_locs.shape}'
        assert self.pb_info_indices.shape == (nb_pboxes,), f'shape error: {self.pb_info_indices.shape}'
        assert self.pb_centers.shape == (nb_pboxes, 2), f'shape error: {self.pb_centers.shape}'
        assert self.pb_sizes.shape == (nb_pboxes, 2), f'shape error: {self.pb_sizes.shape}'
        assert self.pb_mask.shape == (nb_pboxes,), f'shape error: {self.pb_mask.shape}'
        assert len(self.pb_info) == len(self.map_sizes) * len(self.pb_size_patterns)

    def _add_prior_boxes(self, map_sizes, pb_size_patterns):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""
        for map_size in map_sizes:
            # 敷き詰める間隔
            tile_size = 1.0 / map_size
            # 敷き詰めたときの中央の位置のリスト
            lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, map_size, dtype=np.float32)
            # 縦横に敷き詰め
            centers_x, centers_y = np.meshgrid(lin, lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes_center = np.concatenate((centers_x, centers_y), axis=1)
            # (x, y) → タイル×(x1, y1, x2, y2)
            prior_boxes_center = np.tile(prior_boxes_center, (1, 2))
            assert prior_boxes_center.shape == (map_size ** 2, 4)
            # グリッド
            grid = np.copy(prior_boxes_center)
            grid[:, :2] -= tile_size / 2
            grid[:, 2:] += tile_size / 2 + (1 / np.array(self.image_size))

            for pb_pat in pb_size_patterns:
                # prior boxのサイズの基準
                pb_size = tile_size * pb_pat
                # prior boxのサイズ
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                prior_boxes = np.copy(prior_boxes_center)
                prior_boxes[:, :2] -= pb_size / 2
                prior_boxes[:, 2:] += pb_size / 2 + (1 / np.array(self.image_size))

                self.pb_grid.extend(grid)
                self.pb_locs.extend(prior_boxes)
                self.pb_centers.extend(prior_boxes_center[:, :2])
                self.pb_sizes.extend([pb_size] * len(prior_boxes))
                self.pb_info_indices.extend([len(self.pb_info)] * len(prior_boxes))
                self.pb_info.append({
                    'map_size': map_size,
                    'size': np.sqrt(pb_size.prod()),
                    'aspect_ratio': pb_size[0] / pb_size[1],
                    'count': len(prior_boxes),
                })

    def summary(self, logger=None):
        """サマリ表示。"""
        logger = logger or tk.log.get(__name__)
        logger.info('mean objects / image = %f', self.mean_objets)
        logger.info('prior box size ratios = %s', str(self.pb_size_ratios))
        logger.info('prior box aspect ratios = %s', str(self.pb_aspect_ratios))
        logger.info('prior box sizes = %s', str(np.unique([c['size'] for c in self.pb_info])))
        logger.info('prior box count = %d (valid=%d)', len(self.pb_mask), np.count_nonzero(self.pb_mask))

    def encode_locs(self, bboxes, bb_ix, pb_ix):
        """座標を学習用に変換。"""
        if True:
            # x1, y1, x2, y2の回帰
            return (bboxes[bb_ix, :] - self.pb_locs[pb_ix, :]) / np.tile(self.pb_sizes[pb_ix, :], 2) / _VAR_LOC
        else:
            # SSD風のx, y, w, hの回帰
            import keras.backend as K
            bb_center_xy = np.mean([bboxes[bb_ix, :2], bboxes[bb_ix, 2:]], axis=0)
            bb_wh = bboxes[bb_ix, 2:] - bboxes[bb_ix, :2]
            encoded_xy = (bb_center_xy - self.pb_centers[pb_ix, :]) / self.pb_sizes[pb_ix, :] / _VAR_LOC
            encoded_wh = np.log(np.maximum(bb_wh / self.pb_sizes[pb_ix, :], K.epsilon())) / _VAR_SIZE
            return np.concatenate([encoded_xy, encoded_wh], axis=-1)

    def decode_locs(self, pred, xp):
        """encode_locsの逆変換。xpはnumpy or keras.backend。"""
        if True:
            # x1, y1, x2, y2の回帰
            decoded = pred * (_VAR_LOC * np.tile(self.pb_sizes, 2)) + self.pb_locs
            return xp.clip(decoded, 0, 1)
        else:
            # SSD風のx, y, w, hの回帰
            decoded_xy = pred[:, :, :2] * _VAR_LOC * self.pb_sizes + self.pb_centers
            decoded_half_wh = xp.exp(pred[:, :, 2:] * _VAR_SIZE) * self.pb_sizes / 2
            decoded = xp.concatenate([
                decoded_xy - decoded_half_wh,
                decoded_xy + decoded_half_wh,
            ], axis=-1)
            return xp.clip(decoded, 0, 1)

    def check_prior_boxes(self, y_test: [tk.ml.ObjectsAnnotation], class_names):
        """データに対してprior boxがどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        total_errors = 0
        iou_list = []
        assigned_counts = np.zeros((len(self.pb_info),))
        assigned_count_list = []
        unrec_widths = []  # iou < 0.5の横幅
        unrec_heights = []  # iou < 0.5の高さ
        unrec_ars = []  # iou < 0.5のアスペクト比
        rec_delta_locs = []  # Δlocs

        total_gt_boxes = sum([np.sum(np.logical_not(y.difficults)) for y in y_test])

        for y in tk.tqdm(y_test, desc='check_prior_boxes'):
            # 割り当ててみる
            assigned_pb_list, assigned_gt_list, assigned_iou_list = self._assign_boxes(y.bboxes)

            # 1画像あたり何件のprior boxにassignされたか
            assigned_count_list.append(len(assigned_pb_list))
            # 初期の座標のずれ具合の集計
            for assigned_pb, assigned_gt in zip(assigned_pb_list, assigned_gt_list):
                rec_delta_locs.append(self.encode_locs(y.bboxes, assigned_gt, assigned_pb))
            # オブジェクトごとの集計
            for gt_ix, (class_id, difficult) in enumerate(zip(y.classes, y.difficults)):
                assert 1 <= class_id < self.nb_classes
                if difficult:
                    continue
                gt_mask = assigned_gt_list == gt_ix
                if gt_mask.any():
                    gt_iou = assigned_iou_list[gt_mask]
                    gt_pb = assigned_pb_list[gt_mask]
                    max_iou = gt_iou.max()
                    pb_ix = gt_pb[gt_iou.argmax()]
                    # prior box毎の、割り当てられた回数
                    assigned_counts[self.pb_info_indices[pb_ix]] += 1
                else:
                    max_iou = 0
                    # 割り当て失敗
                    total_errors += 1
                # 最大IoU
                iou_list.append(max_iou)
                # 再現率
                y_true.append(class_id)
                y_pred.append(class_id if max_iou >= 0.5 else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                # iou < 0.5しか無いboxの情報を集める
                if max_iou < 0.5:
                    bbox = y.bboxes[gt_ix]
                    wh = bbox[2:] - bbox[:2]
                    unrec_widths.append(wh[1])
                    unrec_heights.append(wh[0])
                    unrec_ars.append(wh[0] / wh[1])
        y_true.append(0)  # 警告よけにbgも1個入れておく
        y_pred.append(0)  # 警告よけにbgも1個入れておく
        cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)

        # ログ出力
        logger = tk.log.get(__name__)
        # ヒストグラム色々
        rec_mean_abs_delta = [np.mean(np.abs(dl)) for dl in rec_delta_locs]
        tk.math.print_histgram(unrec_widths, name='unrec_widths', print_fn=logger.info)
        tk.math.print_histgram(unrec_heights, name='unrec_heights', print_fn=logger.info)
        tk.math.print_histgram(unrec_ars, name='unrec_ars', print_fn=logger.info)
        tk.math.print_histgram(rec_mean_abs_delta, name='rec_mean_abs_delta', print_fn=logger.info)
        tk.math.print_histgram(assigned_count_list, name='assigned_count', print_fn=logger.info)
        # classification_report
        logger.info(cr)
        # prior box毎の集計など
        logger.info('assigned counts:')
        for i, c in enumerate(assigned_counts):
            logger.info('  prior boxes{m=%d, size=%.2f ar=%.2f} = %d (%.02f%%)',
                        self.pb_info[i]['map_size'],
                        self.pb_info[i]['size'],
                        self.pb_info[i]['aspect_ratio'],
                        c, 100 * c / self.pb_info[i]['count'] / total_gt_boxes)
        logger.info('total errors: %d / %d (%.02f%%)',
                    total_errors, total_gt_boxes, 100 * total_errors / total_gt_boxes)
        # iou < 0.5の出現率
        logger.info('[iou < 0.5] count: %d / %d (%.02f%%)',
                    len(unrec_widths), len(y_test), 100 * len(unrec_widths) / len(y_test))
        # YOLOv2の論文のTable 1相当の値のつもり (複数のprior boxに割り当てたときの扱いがちょっと違いそう)
        logger.info('Avg IOU: %.1f', np.mean(iou_list) * 100)
        # 1画像あたり何件のprior boxにassignされたか
        logger.info('assigned count per image: mean=%.1f std=%.2f min=%d max=%d',
                    np.mean(assigned_count_list), np.std(assigned_count_list),
                    min(assigned_count_list), max(assigned_count_list))
        # Δlocの分布調査
        # mean≒0, std≒1とかくらいが学習しやすいはず。(SSDを真似た謎の0.1で大体そうなってる)
        delta_locs = np.concatenate(rec_delta_locs)
        logger.info('delta loc: mean=%.2f std=%.2f min=%.2f max=%.2f',
                    delta_locs.mean(), delta_locs.std(), delta_locs.min(), delta_locs.max())

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
            assigned_pb_list, assigned_gt_list, _ = self._assign_boxes(y.bboxes)
            for pb_ix, gt_ix in zip(assigned_pb_list, assigned_gt_list):
                class_id = y.classes[gt_ix]
                assert 0 < class_id < self.nb_classes
                confs[i, pb_ix, 0] = 0  # bg
                confs[i, pb_ix, class_id] = 1
                locs[i, pb_ix, :] = self.encode_locs(y.bboxes, gt_ix, pb_ix)

        # いったんくっつける (損失関数の中で分割して使う)
        dummy = np.zeros((len(y_gt), len(self.pb_locs), 1), dtype=np.float32)  # ネットワークの出力と形式を合わせる
        return np.concatenate([confs, locs, dummy], axis=-1)

    def _assign_boxes(self, bboxes):
        """各bounding boxをprior boxに割り当てる。

        戻り値は、prior boxのindexとbboxesのindexとiouのタプルのリスト。
        """
        assert len(bboxes) >= 1
        bb_centers = np.mean([bboxes[:, 2:], bboxes[:, :2]], axis=0)

        assigned_gt = -np.ones((len(self.pb_locs),), dtype=int)  # -1埋め
        assigned_iou = np.zeros((len(self.pb_locs),), dtype=float)
        assignable = np.ones((len(self.pb_locs),), dtype=bool)

        # 面積の降順に並べ替え (おまじない: 出来るだけ小さいのが埋もれないように)
        sorted_indices = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=-1).argsort()[::-1]

        for gt_ix, bbox, bb_center in zip(sorted_indices, bboxes[sorted_indices], bb_centers[sorted_indices]):
            # bboxの重心が含まれるprior boxにのみ割り当てる。
            pb_center_mask = np.logical_and(self.pb_grid[:, :2] <= bb_center, bb_center < self.pb_grid[:, 2:]).all(axis=-1)
            pb_center_mask = np.logical_and(pb_center_mask, self.pb_mask)
            assert pb_center_mask.any(), f'Encode error: {bb_center}'
            # IoUが0.5以上のものに割り当てる。1つも無ければ最大のものに。
            iou = tk.ml.compute_size_based_iou(np.expand_dims(bbox, axis=0), self.pb_locs[pb_center_mask, :])[0]
            iou_mask = iou >= 0.5
            if iou_mask.any():
                for pb_ix, pb_iou in zip(np.where(pb_center_mask)[0][iou_mask], iou[iou_mask]):
                    # よりIoUが大きいものを優先して割り当て
                    if assignable[pb_ix] and assigned_iou[pb_ix] < pb_iou:
                        assigned_gt[pb_ix] = gt_ix
                        assigned_iou[pb_ix] = pb_iou
            else:
                iou_mask = iou.argmax()
                pb_mask = np.where(pb_center_mask)[0][iou_mask]
                assigned_gt[pb_mask] = gt_ix
                assigned_iou[pb_mask] = iou[iou_mask]
                assignable[pb_mask] = False  # 上書き禁止！

        assert not np.logical_and(assigned_gt >= 0, np.logical_not(self.pb_mask)).any(), '無効なprior boxへの割り当て'

        pb_indices = np.where(assigned_gt >= 0)[0]
        assert len(pb_indices) >= 1  # 1個以上は必ず割り当てないと損失関数などが面倒になる

        return pb_indices, assigned_gt[pb_indices], assigned_iou[pb_indices]

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
            jobs = [joblib.delayed(self._select, check_pickle=False)(pc, pf, pl, top_k, nms_threshold)
                    for pc, pf, pl in zip(classes_list, confs_list, locs_list)]
            result_classes, result_confs, result_locs = zip(*parallel(jobs))
        else:
            result_classes = []
            result_confs = []
            result_locs = []
            for pc, pf, pl in zip(classes_list, confs_list, locs_list):
                img_classes, img_confs, img_locs = self._select(pc, pf, pl, top_k, nms_threshold)
                result_classes.append(img_classes)
                result_confs.append(img_confs)
                result_locs.append(img_locs)
        assert len(result_classes) == len(classes_list)
        assert len(result_confs) == len(classes_list)
        assert len(result_locs) == len(classes_list)
        return result_classes, result_confs, result_locs

    def _select(self, pred_classes, pred_confs, pred_locs, top_k, nms_threshold):
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
        gt_confs, gt_locs = y_true[:, :, :-5], y_true[:, :, -5:-1]
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
        import keras.backend as K

        def loss_conf(y_true, y_pred):
            """クラス分類の損失項。(metrics用)"""
            gt_confs = y_true[:, :, :-5]
            pred_confs = y_pred[:, :, :-5]
            obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
            obj_count = K.sum(obj_mask, axis=-1)
            return self._loss_conf(gt_confs, pred_confs, obj_count)

        def loss_loc(y_true, y_pred):
            """位置の損失項。(metrics用)"""
            gt_confs, gt_locs = y_true[:, :, :-5], y_true[:, :, -5:-1]
            pred_locs = y_pred[:, :, -5:-1]
            obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
            obj_count = K.sum(obj_mask, axis=-1)
            return self._loss_loc(gt_locs, pred_locs, obj_mask, obj_count)

        def loss_iou(y_true, y_pred):
            """IOUの損失項。(metrics用)"""
            gt_confs, gt_locs = y_true[:, :, :-5], y_true[:, :, -5:-1]
            pred_locs, pred_iou = y_pred[:, :, -5:-1], y_pred[:, :, -1]
            obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
            obj_count = K.sum(obj_mask, axis=-1)
            return self._loss_iou(gt_locs, pred_locs, pred_iou, obj_mask, obj_count)

        def acc_bg(y_true, y_pred):
            """背景の再現率。"""
            gt_confs = y_true[:, :, :-5]
            pred_confs = y_pred[:, :, :-5]
            bg_mask = K.cast(K.greater_equal(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景
            bg_count = K.sum(bg_mask, axis=-1)
            acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
            return K.sum(acc * bg_mask, axis=-1) / bg_count

        def acc_obj(y_true, y_pred):
            """物体の再現率。"""
            gt_confs = y_true[:, :, :-5]
            pred_confs = y_pred[:, :, :-5]
            obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
            obj_count = K.sum(obj_mask, axis=-1)
            acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
            return K.sum(acc * obj_mask, axis=-1) / obj_count

        return [loss_conf, loss_loc, loss_iou, acc_bg, acc_obj]

    def _loss_conf(self, gt_confs, pred_confs, obj_count):
        """分類のloss。"""
        import keras.backend as K
        if True:
            loss = tk.dl.losses.categorical_focal_loss(gt_confs, pred_confs)
            loss *= np.expand_dims(self.pb_mask, axis=0)
            loss = K.sum(loss, axis=-1) / obj_count  # normalized by the number of anchors assigned to a ground-truth box
        else:
            loss = tk.dl.losses.categorical_crossentropy(gt_confs, pred_confs, alpha=0.75)
            loss *= np.expand_dims(self.pb_mask, axis=0)
            loss = K.mean(loss, axis=-1)
        return loss

    @staticmethod
    def _loss_loc(gt_locs, pred_locs, obj_mask, obj_count):
        """位置のloss。"""
        import keras.backend as K
        loss = tk.dl.losses.l1_smooth_loss(gt_locs, pred_locs)
        loss = K.sum(loss * obj_mask, axis=-1) / obj_count  # mean
        return loss

    def _loss_iou(self, gt_locs, pred_locs, pred_iou, obj_mask, obj_count):
        """IOUのloss。

        gt_locsとpred_locsからIOUを算出し、それとpred_iouのbinary crossentropyを返す。
        """
        import keras.backend as K
        bboxes_a = self.decode_locs(gt_locs, K)
        bboxes_b = self.decode_locs(pred_locs, K)
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

    def get_preprocess_input(self):
        """`preprocess_input`を返す。"""
        if self.base_network in ('vgg16', 'resnet50'):
            return tk.image.preprocess_input_mean
        else:
            assert self.base_network in ('custom', 'xception')
            return tk.image.preprocess_input_abs1

    @tk.log.trace()
    def create_pretrain_network(self, image_size):
        """事前学習用モデルの作成。"""
        import keras
        builder = tk.dl.layers.Builder()

        x = inputs = keras.layers.Input(image_size + (3,))
        x, _ = self._create_basenet(builder, x, load_weights=True)
        assert builder.shape(x)[1] == 3
        x = builder.conv2d(2048, (1, 1), use_bn=False, use_act=False, name='tail_for_xception')(x)

        model = keras.models.Model(inputs=inputs, outputs=x)

        logger = tk.log.get(__name__)
        logger.info('network depth: %d', tk.dl.models.count_network_depth(model))
        logger.info('trainable params: %d', tk.dl.models.count_trainable_params(model))

        return model

    @tk.log.trace()
    def create_network(self, load_weights=True, for_predict=False):
        """モデルの作成。"""
        import keras
        builder = tk.dl.layers.Builder()

        # downsampling (ベースネットワーク)
        x = inputs = keras.layers.Input(self.image_size + (3,))
        x, ref = self._create_basenet(builder, x, load_weights)
        map_size = builder.shape(x)[1]

        # center
        x = builder.conv2d(256, (3, 3), name='center_conv1')(x)
        x = builder.conv2d(256, (3, 3), name='center_conv2')(x)
        x = keras.layers.AveragePooling2D((map_size, map_size), name='center_ds')(x)
        x = builder.conv2d(256, (1, 1), name='center_dense')(x)
        ref[f'out{1}'] = x

        # upsampling
        while True:
            in_map_size = builder.shape(x)[1]
            assert map_size % in_map_size == 0, f'map size error: {in_map_size} -> {map_size}'
            up_size = map_size // in_map_size
            x = builder.conv2dtr(256, (up_size, up_size), strides=(up_size, up_size), padding='valid',
                                 kernel_initializer='zeros',
                                 use_bias=False, use_bn=False, use_act=False,
                                 name=f'up{map_size}_us')(x)
            t = ref[f'down{map_size}']
            t = builder.conv2d(256, (1, 1), use_act=False, name=f'up{map_size}_lt')(t)
            x = keras.layers.add([x, t], name=f'up{map_size}_mix')
            x = builder.bn(name=f'up{map_size}_mix_bn')(x)
            x = builder.act(name=f'up{map_size}_mix_act')(x)
            x = builder.conv2d(256, (3, 3), name=f'up{map_size}_conv1')(x)
            x = builder.conv2d(256, (3, 3), name=f'up{map_size}_conv2')(x)
            ref[f'out{map_size}'] = x

            if self.map_sizes[0] <= map_size:
                break
            map_size *= 2

        # prediction module
        confs, locs, ious = self._create_pm(builder, ref)

        if for_predict:
            model = self._create_predict_network(inputs, confs, locs, ious)
        else:
            # いったんくっつける (損失関数の中で分割して使う)
            outputs = keras.layers.concatenate([confs, locs, ious], axis=-1, name='outputs')
            model = keras.models.Model(inputs=inputs, outputs=outputs)

        logger = tk.log.get(__name__)
        logger.info('network depth: %d', tk.dl.models.count_network_depth(model))
        logger.info('trainable params: %d', tk.dl.models.count_trainable_params(model))

        return model

    @tk.log.trace()
    def _create_basenet(self, builder, x, load_weights):
        """ベースネットワークの作成。"""
        import keras

        def _freeze(model, freeze_end_layer):
            for layer in model.layers:
                if layer.name == freeze_end_layer:
                    break
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False

        ref_list = []
        if self.base_network == 'custom':
            x = builder.conv2d(32, (7, 7), strides=(2, 2), name='stage0_ds')(x)
            x = keras.layers.MaxPooling2D(name='stage1_ds')(x)
            x = builder.conv2d(64, (3, 3), name='stage2_conv1')(x)
            x = builder.conv2d(64, (3, 3), name='stage2_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage2_ds')(x)
            x = builder.conv2d(128, (3, 3), name='stage3_conv1')(x)
            x = builder.conv2d(128, (3, 3), name='stage3_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage3_ds')(x)
            x = builder.conv2d(256, (3, 3), name='stage4_conv1')(x)
            x = builder.conv2d(256, (3, 3), name='stage4_conv2')(x)
            ref_list.append(x)
        elif self.base_network == 'vgg16':
            basenet = keras.applications.VGG16(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            _freeze(basenet, 'block5_conv1')
            ref_list.append(basenet.get_layer(name='block4_pool').input)
            ref_list.append(basenet.get_layer(name='block5_pool').input)
        elif self.base_network == 'resnet50':
            basenet = keras.applications.ResNet50(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            _freeze(basenet, 'res4f_branch2a')
            ref_list.append(basenet.get_layer(name='res4a_branch2a').input)
            ref_list.append(basenet.get_layer(name='res5a_branch2a').input)
            ref_list.append(basenet.get_layer(name='avg_pool').input)
        elif self.base_network == 'xception':
            basenet = keras.applications.Xception(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            _freeze(basenet, 'block10_sepconv1_act')
            ref_list.append(basenet.get_layer(name='block4_sepconv1_act').input)
            ref_list.append(basenet.get_layer(name='block13_sepconv1_act').input)
            ref_list.append(basenet.get_layer(name='block14_sepconv2_act').output)
        else:
            assert False

        x = ref_list[-1]
        if builder.shape(x)[-1] > 256:
            x = builder.conv2d(256, (1, 1), name='tail_sq')(x)
        assert builder.shape(x)[-1] == 256

        # downsampling
        while True:
            map_size = builder.shape(x)[1] // 2
            x = builder.conv2d(256, (2, 2), strides=(2, 2), name=f'down{map_size}_ds')(x)
            x = builder.conv2d(256, (3, 3), name=f'down{map_size}_conv')(x)
            assert builder.shape(x)[1] == map_size
            ref_list.append(x)
            if map_size <= 4 or map_size % 2 != 0:  # 充分小さくなるか奇数になったら終了
                break

        ref = {f'down{builder.shape(x)[1]}': x for x in ref_list}
        return x, ref

    @tk.log.trace()
    def _create_pm(self, builder, ref):
        """Prediction module."""
        import keras

        old_gn = builder.use_gn
        builder.use_gn = True  # TODO: いまいち…

        shared_layers = {}
        for pat_ix in range(len(self.pb_size_patterns)):
            shared_layers[f'pm-{pat_ix}_conv1'] = builder.conv2d(64, (1, 1), name=f'pm-{pat_ix}_conv1')
            shared_layers[f'pm-{pat_ix}_conv2'] = builder.conv2d(64, (3, 3), name=f'pm-{pat_ix}_conv2')
            shared_layers[f'pm-{pat_ix}_conf'] = builder.conv2d(
                self.nb_classes, (1, 1),
                kernel_initializer='zeros',
                bias_initializer=tk.dl.losses.od_bias_initializer(self.nb_classes),
                bias_regularizer=None,
                activation='softmax',
                use_bn=False,
                name=f'pm-{pat_ix}_conf')
            shared_layers[f'pm-{pat_ix}_loc'] = builder.conv2d(
                4, (1, 1),
                kernel_initializer='zeros',
                use_bn=False,
                use_act=False,
                name=f'pm-{pat_ix}_loc')
            shared_layers[f'pm-{pat_ix}_iou'] = builder.conv2d(
                1, (1, 1),
                kernel_initializer='zeros',
                activation='sigmoid',
                use_bn=False,
                name=f'pm-{pat_ix}_iou')

        confs, locs, ious = [], [], []
        for map_size in self.map_sizes:
            assert f'out{map_size}' in ref, f'map_size error: {ref}'
            map_out = ref[f'out{map_size}']
            for pat_ix in range(len(self.pb_size_patterns)):
                x = map_out
                x = shared_layers[f'pm-{pat_ix}_conv1'](x)
                x = shared_layers[f'pm-{pat_ix}_conv2'](x)
                conf = shared_layers[f'pm-{pat_ix}_conf'](x)
                loc = shared_layers[f'pm-{pat_ix}_loc'](x)
                iou = shared_layers[f'pm-{pat_ix}_iou'](x)
                conf = keras.layers.Reshape((-1, self.nb_classes), name=f'pm{map_size}-{pat_ix}_reshape_conf')(conf)
                loc = keras.layers.Reshape((-1, 4), name=f'pm{map_size}-{pat_ix}_reshape_loc')(loc)
                iou = keras.layers.Reshape((-1, 1), name=f'pm{map_size}-{pat_ix}_reshape_iou')(iou)
                confs.append(conf)
                locs.append(loc)
                ious.append(iou)

        confs = keras.layers.concatenate(confs, axis=-2, name='output_confs')
        locs = keras.layers.concatenate(locs, axis=-2, name='output_locs')
        ious = keras.layers.concatenate(ious, axis=-2, name='output_ious')

        builder.use_gn = old_gn

        return confs, locs, ious

    def _create_predict_network(self, inputs, confs, locs, ious):
        """予測用ネットワークの作成"""
        import keras
        import keras.backend as K

        def _class(confs):
            return K.argmax(confs[:, :, 1:], axis=-1) + 1

        def _conf(x):
            confs = K.max(x[0][:, :, 1:], axis=-1)
            ious = x[1][:, :, 0]
            return K.sqrt(confs * ious) * np.expand_dims(self.pb_mask, axis=0)

        def _locs(locs):
            return self.decode_locs(locs, K)

        objclasses = keras.layers.Lambda(_class, K.int_shape(confs)[1:-1])(confs)
        objconfs = keras.layers.Lambda(_conf, K.int_shape(confs)[1:-1])([confs, ious])
        locs = keras.layers.Lambda(_locs, K.int_shape(locs)[1:])(locs)

        return keras.models.Model(inputs=inputs, outputs=[objclasses, objconfs, locs])

    def create_generator(self, encode_truth=True):
        """ImageDataGeneratorを作って返す。"""
        gen = tk.image.ImageDataGenerator()
        gen.add(tk.image.CustomAugmentation(self._transform, probability=1))
        gen.add(tk.image.Resize(self.image_size))
        gen.add(tk.image.RandomColorAugmentors(probability=0.5))
        gen.add(tk.image.RandomErasing(probability=0.5))
        gen.add(tk.image.ProcessInput(self.get_preprocess_input(), batch_axis=True))
        if encode_truth:
            gen.add(tk.image.ProcessOutput(self._process_output))
        return gen

    def _transform(self, rgb: np.ndarray, y: tk.ml.ObjectsAnnotation, w, rand: np.random.RandomState, ctx: tk.generator.GeneratorContext):
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
            rgb = self._padding(rgb, ar, rand, y)
        else:
            # crop (zoom in)
            # SSDでは結構複雑なことをやっているが、とりあえず簡単に実装
            rgb = self._crop(y, rgb, rand, ar)
        return rgb, y, w

    def _padding(self, rgb, ar, rand, y):
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
            rgb = tk.ndimage.resize(rgb, new_w, new_h, padding=None)
            # パディング
            px = int(round(px * rw))
            py = int(round(py * rh))
            padding = rand.choice(('edge', 'zero', 'one', 'rand'))
            rgb = tk.ndimage.pad_ltrb(rgb, px, py, self.image_size[1] - new_w - px, self.image_size[0] - new_h - py, padding, rand)
            assert rgb.shape[1] == self.image_size[1]
            assert rgb.shape[0] == self.image_size[0]
            break
        return rgb

    def _crop(self, y, rgb, rand, ar):
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
        return rgb

    def _process_output(self, y):
        if y is not None:
            y = self.encode_truth([y])[0]
        return y

    def save(self, path: pathlib.Path):
        """保存。"""
        joblib.dump(self, path)

    @staticmethod
    def load(path: pathlib.Path):
        """読み込み。"""
        od = joblib.load(path)  # type: ObjectDetector
        od.summary()
        return od
