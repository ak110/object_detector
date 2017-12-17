"""ObjectDetectionのモデル。"""
import numpy as np
import sklearn.cluster
import sklearn.externals.joblib as joblib
import sklearn.metrics
from tqdm import tqdm

import model_net
import pytoolkit as tk

_VAR_LOC = 0.2  # SSD風適当スケーリング
_VAR_SIZE = 0.2  # SSD風適当スケーリング


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    @classmethod
    def create(cls, input_size, map_sizes, nb_classes, y_train, pb_size_pattern_count=8):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。

        gridに配置したときのIOUを直接最適化するのは難しそうなので、
        とりあえず大雑把にKMeansでクラスタ化したりなど。
        """
        map_sizes = np.array(sorted(map_sizes)[::-1])

        # 平均オブジェクト数
        mean_objets = np.mean([len(y.bboxes) for y in y_train])

        # bboxのサイズ
        bboxes = np.concatenate([y.bboxes for y in y_train])
        bboxes_sizes = bboxes[:, 2:] - bboxes[:, :2]
        bb_base_sizes = bboxes_sizes.min(axis=-1)  # 短辺のサイズのリスト
        bb_base_areas = np.sqrt(bboxes_sizes.prod(axis=-1))  # 縦幅・横幅の相乗平均のリスト

        # bboxのサイズごとにどこかのfeature mapに割り当てたことにして相対サイズをリストアップ
        assert (bb_base_sizes >= 0).all()  # 不正なデータは含まれない想定 (手抜き)
        assert (bb_base_sizes < 1).all()  # 不正なデータは含まれない想定 (手抜き)
        tile_sizes = 1 / map_sizes
        bboxes_size_patterns = []
        for ti, tile_size in enumerate(tile_sizes):
            min_tile_size = 0 if tile_size == tile_sizes.min() else tile_size
            max_tile_size = 1 if tile_size == tile_sizes.max() else tile_sizes[ti + 1]
            mask = np.logical_and(bb_base_sizes >= min_tile_size, bb_base_sizes < max_tile_size)
            mask = np.logical_and(mask, bb_base_areas < 0.8)  # 0.8以上のものは中央専用のprior boxを使用
            bboxes_size_patterns.append(bboxes_sizes[mask] / tile_size)
        bboxes_size_patterns = np.concatenate(bboxes_size_patterns)
        assert len(bboxes_size_patterns.shape) == 2
        assert bboxes_size_patterns.shape[1] == 2

        # リストアップしたものをクラスタリング。(YOLOv2のDimension Clustersのようなもの)。
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_pattern_count, n_jobs=-1)
        pb_size_patterns = cluster.fit(bboxes_size_patterns).cluster_centers_
        assert pb_size_patterns.shape == (pb_size_pattern_count, 2)

        return cls(input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns)

    def __init__(self, input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns):
        self.image_size = (input_size, input_size)
        self.map_sizes = np.array(map_sizes)
        self.nb_classes = nb_classes
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。[1.5, 0.5] なら横が1.5倍、縦が0.5倍のものを用意。
        # 面積昇順に並べ替えておく。
        self.pb_size_patterns = pb_size_patterns[pb_size_patterns.prod(axis=-1).argsort(), :].astype(np.float32)
        # 画像全体用のprior boxのサイズのリスト
        self.pb_center_size_patterns = np.array([[0.8, 0.8], [0.8, 1.0], [1.0, 0.8], [1.0, 1.0]])
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
        self._add_prior_boxes([1], self.pb_center_size_patterns)
        self.pb_grid = np.array(self.pb_grid)
        self.pb_locs = np.array(self.pb_locs)
        self.pb_info_indices = np.array(self.pb_info_indices)
        self.pb_centers = np.array(self.pb_centers)
        self.pb_sizes = np.array(self.pb_sizes)

        # prior boxの重心はpb_gridの中心であるはず
        ct = np.mean([self.pb_locs[:, 2:], self.pb_locs[:, :2]], axis=0)
        assert np.logical_and(self.pb_grid[:, :2] <= ct, ct < self.pb_grid[:, 2:]).all()
        # はみ出ているprior boxは使用しないようにする
        self.pb_mask = np.logical_and(self.pb_grid >= 0, self.pb_grid <= 1 + (1 / self.image_size[0])).all(axis=-1)

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4), 'shape error: {}'.format(self.pb_locs.shape)
        assert self.pb_info_indices.shape == (nb_pboxes,), 'shape error: {}'.format(self.pb_info_indices.shape)
        assert self.pb_centers.shape == (nb_pboxes, 2), 'shape error: {}'.format(self.pb_centers.shape)
        assert self.pb_sizes.shape == (nb_pboxes, 2), 'shape error: {}'.format(self.pb_sizes.shape)
        assert self.pb_mask.shape == (nb_pboxes,), 'shape error: {}'.format(self.pb_mask.shape)
        assert len(self.pb_info) == len(self.map_sizes) * len(self.pb_size_patterns) + len(self.pb_center_size_patterns)

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

    def check_prior_boxes(self, logger, result_dir, y_test: [tk.ml.ObjectsAnnotation], class_names):
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

        for y in tqdm(y_test, desc='check_prior_boxes', ascii=True, ncols=100):
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
        logger.debug(cr)
        logger.debug('assigned counts:')
        for i, c in enumerate(assigned_counts):
            logger.debug('  prior boxes{m=%d, size=%.2f ar=%.2f} = %d (%.02f%%)',
                         self.pb_info[i]['map_size'],
                         self.pb_info[i]['size'],
                         self.pb_info[i]['aspect_ratio'],
                         c, 100 * c / self.pb_info[i]['count'] / total_gt_boxes)
        logger.debug('total errors: %d / %d (%.02f%%)',
                     total_errors, total_gt_boxes, 100 * total_errors / total_gt_boxes)
        # iou < 0.5の出現率
        logger.debug('[iou < 0.5] count: %d / %d (%.02f%%)',
                     len(unrec_widths), len(y_test), 100 * len(unrec_widths) / len(y_test))
        # YOLOv2の論文のTable 1相当の値のつもり (複数のprior boxに割り当てたときの扱いがちょっと違いそう)
        logger.debug('Avg IOU: %.1f', np.mean(iou_list) * 100)
        # 1画像あたり何件のprior boxにassignされたか
        logger.debug('assigned count per image: mean=%.1f std=%.2f min=%d max=%d',
                     np.mean(assigned_count_list), np.std(assigned_count_list),
                     min(assigned_count_list), max(assigned_count_list))
        # Δlocの分布調査
        # mean≒0, std≒1とかくらいが学習しやすいはず。(SSDを真似た謎の0.1で大体そうなってる)
        delta_locs = np.concatenate(rec_delta_locs)
        logger.debug('delta loc: mean=%.2f std=%.2f min=%.2f max=%.2f',
                     delta_locs.mean(), delta_locs.std(), delta_locs.min(), delta_locs.max())

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
        plt.hist(assigned_count_list, bins=32)
        plt.gcf().savefig(str(result_dir.joinpath('assigned_count.hist.png')))
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

        assigned_gt = -np.ones((len(self.pb_locs),), dtype=int)  # -1埋め
        assigned_iou = np.zeros((len(self.pb_locs),), dtype=float)
        assignable = np.ones((len(self.pb_locs),), dtype=bool)

        # 面積の降順に並べ替え (おまじない: 出来るだけ小さいのが埋もれないように)
        sorted_indices = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=-1).argsort()[::-1]

        for gt_ix, bbox in zip(sorted_indices, bboxes[sorted_indices]):
            # bboxの重心が含まれるprior boxにのみ割り当てる。
            bb_center = np.mean([bbox[2:], bbox[:2]], axis=0)
            pb_center_mask = np.logical_and(self.pb_locs[:, :2] <= bb_center, bb_center < self.pb_locs[:, 2:]).all(axis=-1)
            pb_center_mask = np.logical_and(pb_center_mask, self.pb_mask)
            assert pb_center_mask.any(), 'Encode error: {}'.format(bb_center)
            # IoUが0.5以上のものに割り当てる。1つも無ければ最大のものに。
            iou = tk.ml.compute_iou(np.expand_dims(bbox, axis=0), self.pb_locs[pb_center_mask, :])[0]
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
        return [self.loss_conf, self.loss_loc, self.loss_iou, self.acc_bg, self.acc_obj]

    def loss_conf(self, y_true, y_pred):
        """クラス分類の損失項。(metrics用)"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-5]
        pred_confs = y_pred[:, :, :-5]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return self._loss_conf(gt_confs, pred_confs, obj_count)

    @staticmethod
    def loss_loc(y_true, y_pred):
        """位置の損失項。(metrics用)"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-5], y_true[:, :, -5:-1]
        pred_locs = y_pred[:, :, -5:-1]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return ObjectDetector._loss_loc(gt_locs, pred_locs, obj_mask, obj_count)

    def loss_iou(self, y_true, y_pred):
        """IOUの損失項。(metrics用)"""
        import keras.backend as K
        gt_confs, gt_locs = y_true[:, :, :-5], y_true[:, :, -5:-1]
        pred_locs, pred_iou = y_pred[:, :, -5:-1], y_pred[:, :, -1]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        return self._loss_iou(gt_locs, pred_locs, pred_iou, obj_mask, obj_count)

    def _loss_conf(self, gt_confs, pred_confs, obj_count):
        """分類のloss。"""
        import keras.backend as K
        if True:
            loss = tk.dl.categorical_focal_loss(gt_confs, pred_confs)
            loss *= np.expand_dims(self.pb_mask, axis=0)
            loss = K.sum(loss, axis=-1) / obj_count  # normalized by the number of anchors assigned to a ground-truth box
        else:
            loss = tk.dl.categorical_crossentropy(gt_confs, pred_confs, alpha=0.75)
            loss *= np.expand_dims(self.pb_mask, axis=0)
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

    @staticmethod
    def acc_bg(y_true, y_pred):
        """背景の再現率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-5]
        pred_confs = y_pred[:, :, :-5]
        bg_mask = K.cast(K.greater_equal(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景
        bg_count = K.sum(bg_mask, axis=-1)
        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * bg_mask, axis=-1) / bg_count

    @staticmethod
    def acc_obj(y_true, y_pred):
        """物体の再現率。"""
        import keras.backend as K
        gt_confs = y_true[:, :, :-5]
        pred_confs = y_pred[:, :, :-5]
        obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
        obj_count = K.sum(obj_mask, axis=-1)
        acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
        return K.sum(acc * obj_mask, axis=-1) / obj_count

    def create_network(self, base_network='resnet50'):
        """ネットワークの作成"""
        return model_net.create_network(self, base_network)

    def create_predict_network(self, model):
        """予測用ネットワークの作成"""
        return model_net.create_predict_network(self, model)
