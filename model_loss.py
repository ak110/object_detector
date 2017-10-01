"""損失関数とmetrics。"""

import pytoolkit as tk


def multibox_loss(y_true, y_pred):
    """損失関数。"""
    import keras.backend as K
    import tensorflow as tf
    gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
    pred_confs, pred_locs = y_pred[:, :, :-4], y_pred[:, :, -4:]

    obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
    obj_count = K.sum(obj_mask, axis=-1)  # 各batch毎のobj数。

    # obj_countが1以上であることの確認
    with tf.control_dependencies([tf.assert_positive(obj_count)]):
        obj_count = tf.identity(obj_count)

    loss_conf = _loss_conf(gt_confs, pred_confs, obj_count)
    loss_loc = _loss_loc(gt_locs, pred_locs, obj_mask, obj_count)
    return loss_conf + loss_loc


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


def _loss_loc(gt_locs, pred_locs, obj_mask, obj_count):
    """位置のloss。"""
    import keras.backend as K
    loss = tk.dl.l1_smooth_loss(gt_locs, pred_locs)
    loss = K.sum(loss * obj_mask, axis=-1) / obj_count  # mean
    return loss


def multibox_loss_loc(y_true, y_pred):
    """位置の損失項。(metrics用)"""
    import keras.backend as K
    gt_confs, gt_locs = y_true[:, :, :-4], y_true[:, :, -4:]
    pred_locs = y_pred[:, :, -4:]

    obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
    obj_count = K.sum(obj_mask, axis=-1)

    return _loss_loc(gt_locs, pred_locs, obj_mask, obj_count)


def multibox_acc_bg(y_true, y_pred):
    """背景の再現率。"""
    import keras.backend as K
    gt_confs = y_true[:, :, :-4]
    pred_confs = y_pred[:, :, :-4]

    bg_mask = K.cast(K.greater_equal(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景
    bg_count = K.sum(bg_mask, axis=-1)

    acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
    return K.sum(acc * bg_mask, axis=-1) / bg_count


def multibox_acc_obj(y_true, y_pred):
    """物体の再現率。"""
    import keras.backend as K
    gt_confs = y_true[:, :, :-4]
    pred_confs = y_pred[:, :, :-4]

    obj_mask = K.cast(K.less(gt_confs[:, :, 0], 0.5), K.floatx())   # 背景以外
    obj_count = K.sum(obj_mask, axis=-1)

    acc = K.cast(K.equal(K.argmax(gt_confs, axis=-1), K.argmax(pred_confs, axis=-1)), K.floatx())
    return K.sum(acc * obj_mask, axis=-1) / obj_count
