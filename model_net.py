"""ObjectDetectionのモデル。"""
import numpy as np

import pytoolkit as tk

# ベースネットワークの種類: custom, vgg16, resnet50, xception
_BASENET_TYPE = 'resnet50'


def create_network(od, base_network):
    """モデルの作成。"""
    import keras
    import keras.backend as K

    builder = tk.dl.Builder()
    builder.conv_defaults['kernel_initializer'] = 'he_uniform'
    builder.dense_defaults['kernel_initializer'] = 'he_uniform'
    builder.set_default_l2(1e-5)

    # downsampling (ベースネットワーク)
    x = inputs = keras.layers.Input(od.image_size + (3,))
    x, ref, lr_multipliers = _create_basenet(od, builder, x, base_network)
    # center
    map_size = K.int_shape(x)[1]
    x = _centerblock(builder, x, ref, map_size)
    # upsampling
    while True:
        x = _upblock(builder, x, ref, map_size)
        if od.map_sizes[0] <= map_size:
            break
        map_size *= 2
    # prediction module
    outputs = _create_pm(od, builder, ref)

    return keras.models.Model(inputs=inputs, outputs=outputs), lr_multipliers


def get_preprocess_input(base_network):
    """`preprocess_input`を返す。"""
    if base_network in ('vgg16', 'resnet50'):
        return tk.image.preprocess_input_mean
    else:
        assert base_network in ('custom', 'xception')
        return tk.image.preprocess_input_abs1


def _create_basenet(od, builder, x, base_network):
    """ベースネットワークの作成。"""
    import keras
    import keras.backend as K

    ref_list = []

    lr_multipliers = {}
    if base_network == 'custom':
        if K.int_shape(x)[1] < 800:  # 仮：そのうち消すかも
            x = builder.conv2d(32, (7, 7), strides=(2, 2), name='stage0_ds')(x)
            x = builder.conv2d(64, (3, 3), strides=(1, 1), name='stage1_conv')(x)
            x = keras.layers.MaxPooling2D(name='stage1_ds')(x)
            x = _block(builder, x, 256, name='stage2_block')
        else:
            x = builder.conv2d(32, (7, 7), strides=(2, 2), name='stage0_ds')(x)
            x = keras.layers.MaxPooling2D(name='stage1_ds')(x)
            x = builder.conv2d(64, (3, 3), name='stage2_conv1')(x)
            x = builder.conv2d(64, (3, 3), name='stage2_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage2_ds')(x)
            x = builder.conv2d(128, (3, 3), name='stage3_conv1')(x)
            x = builder.conv2d(128, (3, 3), name='stage3_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage3_ds')(x)
            x = _block(builder, x, 256, name='stage4_block')
        ref_list.append(x)
    elif base_network == 'vgg16':
        basenet = keras.applications.VGG16(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [0.01] * len(w)))
        ref_list.append(basenet.get_layer(name='block4_pool').input)
        ref_list.append(basenet.get_layer(name='block5_pool').input)
    elif base_network == 'resnet50':
        basenet = keras.applications.ResNet50(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [0.01] * len(w)))
        ref_list.append(basenet.get_layer(name='res4a_branch2a').input)
        ref_list.append(basenet.get_layer(name='res5a_branch2a').input)
        ref_list.append(basenet.get_layer(name='avg_pool').input)
    elif base_network == 'xception':
        basenet = keras.applications.Xception(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [0.01] * len(w)))
        ref_list.append(basenet.get_layer(name='block4_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block13_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block14_sepconv2_act').output)
    else:
        assert False

    x = ref_list[-1]

    if K.int_shape(x)[-1] > 256:
        x = builder.conv2d(256, (1, 1), name='tail_sq')(x)
    assert K.int_shape(x)[-1] == 256

    # downsampling
    while True:
        x, map_size = _downblock(builder, x)
        ref_list.append(x)
        if map_size <= 4 or map_size % 2 != 0:  # 充分小さくなるか奇数になったら終了
            break

    ref = {'down{}'.format(K.int_shape(x)[1]): x for x in ref_list}

    # 欲しいサイズがちゃんとあるかチェック
    for map_size in od.map_sizes:
        assert 'down{}'.format(map_size) in ref, 'map_size error: {}'.format(ref)

    return x, ref, lr_multipliers


def _block(builder, x, filters, name):
    x = builder.conv2d(filters, (3, 3), name=name + '_c1')(x)
    x = builder.conv2d(filters, (3, 3), name=name + '_c2')(x)
    return x


def _downblock(builder, x):
    import keras
    import keras.backend as K

    map_size = K.int_shape(x)[1] // 2

    x = keras.layers.MaxPooling2D(name='down{}_ds'.format(map_size))(x)
    assert K.int_shape(x)[1] == map_size

    x = _block(builder, x, 256, name='down{}_block'.format(map_size))
    return x, map_size


def _centerblock(builder, x, ref, map_size):
    import keras
    x = _block(builder, x, 256, name='center_block')
    x = keras.layers.AveragePooling2D((map_size, map_size))(x)
    ref['out{}'.format(1)] = x
    return x


def _upblock(builder, x, ref, map_size):
    import keras
    import keras.backend as K

    in_map_size = K.int_shape(x)[1]
    up_size = map_size // in_map_size

    x = keras.layers.UpSampling2D((up_size, up_size), name='up{}_us'.format(map_size))(x)
    t = ref['down{}'.format(map_size)]

    x = builder.conv2d(256, (3, 3), use_bn=False, use_act=False, name='up{}_cx'.format(map_size))(x)
    t = builder.conv2d(256, (1, 1), use_bn=False, use_act=False, bn_kwargs={'gamma_initializer': 'zeros'}, name='up{}_lt'.format(map_size))(t)
    x = keras.layers.Add(name='up{}_mix'.format(map_size))([x, t])
    x = builder.bn(name='up{}_mix_bn'.format(map_size))(x)
    x = builder.act(name='up{}_mix_act'.format(map_size))(x)
    x = builder.conv2d(256, (3, 3), name='up{}_c1'.format(map_size))(x)
    x = builder.conv2d(256, (3, 3), name='up{}_c2'.format(map_size))(x)

    ref['out{}'.format(map_size)] = x
    return x


def _create_pm(od, builder, ref):
    """Prediction module."""
    import keras

    confs, locs, ious = [], [], []
    for map_size in od.map_sizes:
        assert 'out{}'.format(map_size) in ref, 'map_size error: {}'.format(ref)
        x = ref['out{}'.format(map_size)]
        for pat_ix in range(len(od.pb_size_patterns)):
            prefix = 'pm{}-{}'.format(map_size, pat_ix)
            conf, loc, iou = _pm(od, builder, x, prefix)
            confs.append(conf)
            locs.append(loc)
            ious.append(iou)
    for pat_ix in range(len(od.pb_center_size_patterns)):
        prefix = 'pm{}-{}'.format(1, pat_ix)
        conf, loc, iou = _pm_center(od, builder, ref['out{}'.format(1)], prefix)
        confs.append(conf)
        locs.append(loc)
        ious.append(iou)
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)
    ious = keras.layers.Concatenate(axis=-2, name='output_ious')(ious)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs, ious])

    return outputs


def _pm(od, builder, x, prefix):
    import keras
    from keras.regularizers import l2
    # conf/loc/iou
    conf = builder.conv2d(od.nb_classes, (1, 1),
                          kernel_initializer='zeros',
                          bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                          activation='softmax',
                          use_bn=False,
                          use_act=False,
                          name=prefix + '_conf')(x)
    loc = builder.conv2d(4, (1, 1),
                         kernel_initializer='zeros',
                         bias_regularizer=l2(1e-4),  # 平均的には≒0のはず
                         activation=None,
                         use_bn=False,
                         use_act=False,
                         name=prefix + '_loc')(x)
    iou = builder.conv2d(1, (1, 1),
                         kernel_initializer='zeros',
                         activation='sigmoid',
                         use_bn=False,
                         use_act=False,
                         name=prefix + '_iou')(x)
    # reshape
    conf = keras.layers.Reshape((-1, od.nb_classes), name=prefix + '_reshape_conf')(conf)
    loc = keras.layers.Reshape((-1, 4), name=prefix + '_reshape_loc')(loc)
    iou = keras.layers.Reshape((-1, 1), name=prefix + '_reshape_iou')(iou)
    return conf, loc, iou


def _pm_center(od, builder, x, prefix):
    import keras
    from keras.regularizers import l2
    x = keras.layers.Flatten()(x)
    x = builder.dense(128, use_bias=False, name=prefix + '_fc')(x)
    x = builder.bn(name=prefix + '_fc_bn')(x)
    x = builder.act(name=prefix + '_fc_act')(x)
    conf = builder.dense(od.nb_classes,
                         kernel_initializer='zeros',
                         bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                         activation='softmax',
                         name=prefix + '_conf')(x)
    loc = builder.dense(4,
                        kernel_initializer='zeros',
                        bias_regularizer=l2(1e-4),  # 平均的には≒0のはず
                        name=prefix + '_loc')(x)
    iou = builder.dense(1,
                        kernel_initializer='zeros',
                        activation='sigmoid',
                        name=prefix + '_iou')(x)
    conf = keras.layers.Reshape((1, od.nb_classes), name=prefix + '_reshape_conf')(conf)
    loc = keras.layers.Reshape((1, 4), name=prefix + '_reshape_loc')(loc)
    iou = keras.layers.Reshape((1, 1), name=prefix + '_reshape_iou')(iou)
    return conf, loc, iou


def create_predict_network(od, model):
    """予測用ネットワークの作成"""
    import keras
    import keras.backend as K

    try:
        confs = model.get_layer(name='output_confs').output
        locs = model.get_layer(name='output_locs').output
        ious = model.get_layer(name='output_ious').output
    except ValueError:
        output = model.outputs[0]  # マルチGPU対策。。
        confs = keras.layers.Lambda(lambda c: c[:, :, :-5], K.int_shape(output)[1:-1] + (od.nb_classes,))(output)
        locs = keras.layers.Lambda(lambda c: c[:, :, -5:-1], K.int_shape(output)[1:-1] + (4,))(output)
        ious = keras.layers.Lambda(lambda c: c[:, :, -1:], K.int_shape(output)[1:-1] + (1,))(output)

    def _class(confs):
        return K.argmax(confs[:, :, 1:], axis=-1) + 1

    def _conf(x):
        confs = K.max(x[0][:, :, 1:], axis=-1)
        ious = x[1][:, :, 0]
        return K.sqrt(confs * ious) * np.expand_dims(od.pb_mask, axis=0)

    def _locs(locs):
        return od.decode_locs(locs, K)

    objclasses = keras.layers.Lambda(_class, K.int_shape(confs)[1:-1])(confs)
    objconfs = keras.layers.Lambda(_conf, K.int_shape(confs)[1:-1])([confs, ious])
    locs = keras.layers.Lambda(_locs, K.int_shape(locs)[1:])(locs)

    return keras.models.Model(inputs=model.inputs, outputs=[objclasses, objconfs, locs])
