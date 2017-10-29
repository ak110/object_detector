"""ObjectDetectionのモデル。"""
import numpy as np

import pytoolkit as tk

# ベースネットワークの種類: custom, vgg16, resnet50, xception
_BASENET_TYPE = 'resnet50'


def swish(x):
    """Swish: a Self-Gated Activation Function

    https://arxiv.org/abs/1710.05941
    """
    import keras.backend as K
    return x * K.sigmoid(x)


def create_network(od):
    """モデルの作成。"""
    import keras
    import keras.backend as K

    # downsampling (ベースネットワーク)
    x = inputs = keras.layers.Input(od.image_size + (3,))
    x, ref, lr_multipliers = _create_basenet(od, x)
    # center
    map_size = K.int_shape(x)[1]
    x = _centerblock(x, ref, map_size)
    # upsampling
    while True:
        x = _upblock(x, ref, map_size)
        if od.map_sizes[0] <= map_size:
            break
        map_size *= 2
    # prediction module
    outputs = _create_pm(od, ref, lr_multipliers)

    return keras.models.Model(inputs=inputs, outputs=outputs), lr_multipliers


def get_preprocess_input():
    """`preprocess_input`を返す。"""
    if _BASENET_TYPE in ('vgg16', 'resnet50'):
        return tk.image.preprocess_input_mean
    else:
        assert _BASENET_TYPE in ('custom', 'xception')
        return tk.image.preprocess_input_abs1


def _create_basenet(od, x):
    """ベースネットワークの作成。"""
    import keras
    import keras.backend as K

    ref_list = []

    lr_multipliers = {}
    if _BASENET_TYPE == 'custom':
        x = tk.dl.conv2d(32, (7, 7), strides=(2, 2), padding='same', activation=swish, kernel_initializer='he_uniform', name='stage0_ds')(x)  # 160x160
        x = tk.dl.conv2d(64, (3, 3), strides=(1, 1), padding='same', activation=swish, kernel_initializer='he_uniform', name='stage1_conv')(x)
        x = keras.layers.MaxPooling2D(name='stage1_ds')(x)  # 80x80
        x = _denseblock(x, 64, 3, bottleneck=False, compress=False, name='stage2_block')
        ref_list.append(x)
    elif _BASENET_TYPE == 'vgg16':
        basenet = keras.applications.VGG16(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [1 / 100] * len(w)))
        ref_list.append(basenet.get_layer(name='block4_pool').input)
        ref_list.append(basenet.get_layer(name='block5_pool').input)
    elif _BASENET_TYPE == 'resnet50':
        basenet = keras.applications.ResNet50(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [1 / 100] * len(w)))
        ref_list.append(basenet.get_layer(name='res4a_branch2a').input)
        ref_list.append(basenet.get_layer(name='res5a_branch2a').input)
        ref_list.append(basenet.get_layer(name='avg_pool').input)
    elif _BASENET_TYPE == 'xception':
        basenet = keras.applications.Xception(include_top=False, input_tensor=x)
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [1 / 100] * len(w)))
        ref_list.append(basenet.get_layer(name='block4_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block13_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block14_sepconv2_act').output)
    else:
        assert False

    x = ref_list[-1]

    if K.int_shape(x)[-1] > 256:
        x = tk.dl.conv2d(256, (1, 1), activation=swish, kernel_initializer='he_uniform', name='tail_sq')(x)
    assert K.int_shape(x)[-1] == 256
    x = _denseblock(x, 64, 4, bottleneck=True, compress=True, name='tail_block')

    # downsampling
    while True:
        x, map_size = _downblock(x)
        ref_list.append(x)
        if map_size <= 6 or map_size % 2 != 0:  # 充分小さくなるか奇数になったら終了
            break

    ref = {'down{}'.format(K.int_shape(x)[1]): x for x in ref_list}

    # 欲しいサイズがちゃんとあるかチェック
    for map_size in od.map_sizes:
        assert 'down{}'.format(map_size) in ref, 'map_size error: {}'.format(ref)

    return x, ref, lr_multipliers


def _denseblock(x, inc_filters, branches, bottleneck, compress, name):
    import keras
    import keras.backend as K

    for branch in range(branches):
        if bottleneck:
            b = tk.dl.conv2d(inc_filters * 4, (1, 1), padding='same', activation=swish,
                             kernel_initializer='he_uniform',
                             name=name + '_b' + str(branch) + '_c1')(x)
            b = keras.layers.Dropout(0.25)(b)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation=swish,
                             kernel_initializer='he_uniform',
                             name=name + '_b' + str(branch) + '_c2')(b)
        else:
            b = keras.layers.Dropout(0.25)(x)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation=swish,
                             kernel_initializer='he_uniform',
                             name=name + '_b' + str(branch))(b)
        x = keras.layers.Concatenate(name=name + '_b' + str(branch) + '_cat')([x, b])

    if compress:
        x = tk.dl.conv2d(K.int_shape(x)[-1] // 2, (1, 1), padding='same', activation=swish,
                         kernel_initializer='he_uniform', name=name + '_sq')(x)

    return x


def _downblock(x):
    import keras
    import keras.backend as K

    map_size = K.int_shape(x)[1] // 2

    x = keras.layers.MaxPooling2D(name='down{}_ds'.format(map_size))(x)
    assert K.int_shape(x)[1] == map_size

    x = _denseblock(x, 64, 4, bottleneck=True, compress=True, name='down{}_block'.format(map_size))
    return x, map_size


def _centerblock(x, ref, map_size):
    import keras
    x = _denseblock(x, 64, 8, bottleneck=True, compress=True, name='center_block')
    x = keras.layers.AveragePooling2D((map_size, map_size))(x)
    ref['out{}'.format(1)] = x
    return x


def _upblock(x, ref, map_size):
    import keras
    import keras.backend as K

    in_map_size = K.int_shape(x)[1]
    up_size = map_size // in_map_size

    x = keras.layers.UpSampling2D((up_size, up_size), name='up{}_us'.format(map_size))(x)
    t = ref['down{}'.format(map_size)]

    x = tk.dl.conv2d(256, (3, 3), padding='same', activation=None, kernel_initializer='he_uniform', name='up{}_cx'.format(map_size))(x)
    t = tk.dl.conv2d(256, (1, 1), padding='same', activation=None, kernel_initializer='he_uniform', name='up{}_lt'.format(map_size))(t)
    x = keras.layers.Add(name='up{}_mix'.format(map_size))([x, t])
    x = keras.layers.BatchNormalization(name='up{}_mix_bn'.format(map_size))(x)
    x = keras.layers.Activation(activation=swish, name='up{}_mix_act'.format(map_size))(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name='up{}_c1'.format(map_size))(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name='up{}_c2'.format(map_size))(x)

    ref['out{}'.format(map_size)] = x
    return x


def _create_pm(od, ref, lr_multipliers):
    """Prediction module."""
    import keras

    # スケール間で重みを共有するレイヤー
    shared_layers = _pm_create_shared_layers(od)

    confs, locs, ious = [], [], []
    for map_size in od.map_sizes:
        assert 'out{}'.format(map_size) in ref, 'map_size error: {}'.format(ref)
        x = ref['out{}'.format(map_size)]
        for pat_ix in range(len(od.pb_size_patterns)):
            prefix = 'pm{}-{}'.format(map_size, pat_ix)
            conf, loc, iou = _pm(od, x, prefix, shared_layers[pat_ix])
            confs.append(conf)
            locs.append(loc)
            ious.append(iou)
    for pat_ix in range(len(od.pb_center_size_patterns)):
        prefix = 'pm{}-{}'.format(1, pat_ix)
        conf, loc, iou = _pm_center(od, ref['out{}'.format(1)], prefix)
        confs.append(conf)
        locs.append(loc)
        ious.append(iou)
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)
    ious = keras.layers.Concatenate(axis=-2, name='output_ious')(ious)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs, ious])

    # 学習率の調整
    m = [1 / len(od.map_sizes)]
    for shlayers in shared_layers.values():
        for layer in shlayers.values():
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, m * len(w)))
    return outputs


def _pm_create_shared_layers(od):
    import keras
    from keras.regularizers import l2
    shared_layers = {}
    for pat_ix in range(len(od.pb_size_patterns)):
        prefix = 'pm-{}'.format(pat_ix)
        shared_layers[pat_ix] = {
            'sq': keras.layers.Conv2D(64, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name=prefix + '_sq'),
            'c0': keras.layers.Conv2D(32, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name=prefix + '_c0'),
            'c1': keras.layers.Conv2D(32, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name=prefix + '_c1'),
            'c2': keras.layers.Conv2D(32, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name=prefix + '_c2'),
            'c3': keras.layers.Conv2D(32, (3, 3), padding='same', activation=swish, kernel_initializer='he_uniform', name=prefix + '_c3'),
            'conf': keras.layers.Conv2D(od.nb_classes, (3, 3), padding='same',
                                        kernel_initializer='zeros',
                                        kernel_regularizer=l2(1e-4),
                                        bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                                        activation='softmax',
                                        name=prefix + '_conf'),
            'loc': keras.layers.Conv2D(4, (3, 3), padding='same', use_bias=False,  # 平均的には≒0のはずなのでバイアス無し
                                       kernel_initializer='zeros',
                                       kernel_regularizer=l2(1e-4),
                                       name=prefix + '_loc'),
            'iou': keras.layers.Conv2D(1, (3, 3), padding='same',
                                       kernel_initializer='zeros',
                                       kernel_regularizer=l2(1e-4),
                                       activation='sigmoid',
                                       name=prefix + '_iou'),
        }
    return shared_layers


def _pm(od, x, prefix, shlayers):
    import keras
    # squeeze
    x = shlayers['sq'](x)
    # DenseBlock
    for branch in range(4):
        b = keras.layers.Dropout(0.25, name='{}_c{}_drop'.format(prefix, branch))(x)
        b = shlayers['c' + str(branch)](b)
        x = keras.layers.Concatenate(name='{}_c{}_cat'.format(prefix, branch))([x, b])
    # conf/loc/iou
    conf = shlayers['conf'](x)
    loc = shlayers['loc'](x)
    mix = keras.layers.Concatenate(name=prefix + '_mix')([x, conf, loc])
    iou = shlayers['iou'](mix)
    # reshape
    conf = keras.layers.Reshape((-1, od.nb_classes), name=prefix + '_reshape_conf')(conf)
    loc = keras.layers.Reshape((-1, 4), name=prefix + '_reshape_loc')(loc)
    iou = keras.layers.Reshape((-1, 1), name=prefix + '_reshape_iou')(iou)
    return conf, loc, iou


def _pm_center(od, x, prefix):
    import keras
    from keras.regularizers import l2
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation=swish, kernel_initializer='he_uniform',
                           name=prefix + '_fc')(x)
    conf = keras.layers.Dense(od.nb_classes,
                              kernel_initializer='zeros',
                              kernel_regularizer=l2(1e-4),
                              bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                              activation='softmax',
                              name=prefix + '_conf')(x)
    loc = keras.layers.Dense(4, use_bias=False,  # 平均的には≒0のはずなのでバイアス無し
                             kernel_initializer='zeros',
                             kernel_regularizer=l2(1e-4),
                             name=prefix + '_loc')(x)
    mix = keras.layers.Concatenate(name=prefix + '_mix')([x, conf, loc])
    iou = keras.layers.Dense(1,
                             kernel_initializer='zeros',
                             kernel_regularizer=l2(1e-4),
                             activation='sigmoid',
                             name=prefix + '_iou')(mix)
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
