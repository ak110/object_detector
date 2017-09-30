"""ObjectDetectionのモデル。"""

import pytoolkit as tk

# ベースネットワークの種類: custom, resnet50, xception
_BASENET_TYPE = 'resnet50'


def create_pretrain_network(input_shape, nb_classes):
    """モデルの作成。"""
    import keras

    # ベースネットワーク
    x = inputs = keras.layers.Input(input_shape)
    x, _ = _create_basenet(x)

    # center
    x = _downblock(x, 'center_block')

    # prediction
    x = keras.layers.GlobalAveragePooling2D(name='pool')(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2', name='pretrain_predictions')(x)

    return keras.models.Model(inputs=inputs, outputs=x)


def create_network(od, freeze):
    """モデルの作成。"""
    import keras
    from model import ObjectDetector

    # ベースネットワーク
    x = inputs = keras.layers.Input(od.input_size + (3,))
    x, ref = _create_basenet(x, freeze)

    # center
    x = _downblock(x, 'center_block')

    # upsampling
    for fm_count in ObjectDetector.FM_COUNTS[::-1]:
        if fm_count != ObjectDetector.FM_COUNTS[-1]:
            x = keras.layers.UpSampling2D(name='up{}_us'.format(fm_count))(x)
        x = _upblock(x, ref['down{}'.format(fm_count)], 'up{}_block'.format(fm_count))
        ref['up{}'.format(fm_count)] = x

    # prediction module
    confs, locs = _create_pm(od, ref)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    return keras.models.Model(inputs=inputs, outputs=outputs)


def get_input_size():
    """入力画像サイズ。"""
    if _BASENET_TYPE == 'custom':
        return (325, 325)
    else:
        assert _BASENET_TYPE in ('resnet50', 'xception')
        return (319, 319)


def get_preprocess_input():
    """`preprocess_input`を返す。"""
    if _BASENET_TYPE == 'resnet50':
        return tk.image.preprocess_input_mean
    else:
        assert _BASENET_TYPE in ('custom', 'xception')
        return tk.image.preprocess_input_abs1


def _create_basenet(x, freeze):
    """ベースネットワークの作成。"""
    import keras
    import keras.backend as K
    from model import ObjectDetector

    ref = {}
    if _BASENET_TYPE == 'custom':
        x = tk.dl.conv2d(32, (7, 7), strides=(2, 2), padding='valid', activation='relu', name='stage1_conv1')(x)  # 160x160
        x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='stage1_conv2')(x)
        x = keras.layers.MaxPooling2D(name='stage1_ds')(x)  # 80x80。DenseNetを参考に最初だけMax。
        x = _denseblock(x, 64, 3, bottleneck=False, compress=False, name='stage2_block')
        for fm_count in ObjectDetector.FM_COUNTS:
            x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
            x = _downblock(x, 'down{}_block'.format(fm_count))
            ref['down{}'.format(fm_count)] = x
    elif _BASENET_TYPE in 'resnet50':
        basenet = keras.applications.ResNet50(include_top=False, input_tensor=x)
        if freeze:
            for layer in basenet.layers:
                layer.trainable = False
        ref['down{}'.format(40)] = basenet.get_layer(name='res4a_branch2a').input
        ref['down{}'.format(20)] = basenet.get_layer(name='res5a_branch2a').input
        ref['down{}'.format(10)] = basenet.get_layer(name='avg_pool').input
        x = ref['down{}'.format(10)]
        x = tk.dl.conv2d(512, (1, 1), activation='relu', name='pre_squeeze')(x)
        for fm_count in (5,):
            x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
            x = _downblock(x, 'down{}_block'.format(fm_count))
            ref['down{}'.format(fm_count)] = x
    elif _BASENET_TYPE in 'xception':
        basenet = keras.applications.Xception(include_top=False, input_tensor=x)
        if freeze:
            for layer in basenet.layers:
                layer.trainable = False
        ref['down{}'.format(40)] = basenet.get_layer(name='block4_sepconv1_act').input
        ref['down{}'.format(20)] = basenet.get_layer(name='block13_sepconv1_act').input
        ref['down{}'.format(10)] = basenet.get_layer(name='block14_sepconv2_act').output
        x = ref['down{}'.format(10)]
        x = tk.dl.conv2d(512, (1, 1), activation='relu', name='pre_squeeze')(x)
        for fm_count in (5,):
            x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
            x = _downblock(x, 'down{}_block'.format(fm_count))
            ref['down{}'.format(fm_count)] = x
    else:
        assert False

    for fm_count in ObjectDetector.FM_COUNTS:
        assert 'down{}'.format(fm_count) in ref
        x = ref['down{}'.format(fm_count)]
        assert K.int_shape(x)[1] == fm_count

    return x, ref


def _denseblock(x, inc_filters, branches, bottleneck, compress, name):
    import keras
    import keras.backend as K
    for branch in range(branches):
        if bottleneck:
            b = tk.dl.conv2d(inc_filters * 4, (1, 1), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c1')(x)
            b = keras.layers.Dropout(0.25)(b)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c2')(b)
        else:
            b = keras.layers.Dropout(0.25)(x)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='relu', name=name + '_b' + str(branch))(b)
        x = keras.layers.Concatenate(name=name + '_b' + str(branch) + '_concat')([x, b])
    if compress:
        x = tk.dl.conv2d(K.int_shape(x)[-1] // 2, (1, 1), padding='same', activation='relu', name=name + '_sq')(x)
    return x


def _branch(x, filters, name):
    import keras
    x = tk.dl.conv2d(filters, (3, 3), padding='same', activation='relu', name=name + '_c1')(x)
    x = keras.layers.Dropout(0.25, name=name + '_drop')(x)
    x = tk.dl.conv2d(filters, (3, 3), padding='same', activation='relu', name=name + '_c2')(x)
    return x


def _downblock(x, name):
    x = _denseblock(x, 64, 8, bottleneck=True, compress=True, name=name)
    return x


def _upblock(x, b, name):
    import keras
    import keras.backend as K
    if K.int_shape(x)[-1] != 256:
        x = tk.dl.conv2d(256, (1, 1), padding='same', activation=None, use_bn=False, use_bias=False, name=name + '_sqx')(x)
    if K.int_shape(b)[-1] != 256:
        b = tk.dl.conv2d(256, (1, 1), padding='same', activation=None, use_bn=False, use_bias=False, name=name + '_sqb')(b)
    x = keras.layers.Add(name=name + '_add')([x, b])
    x = keras.layers.BatchNormalization(name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_act')(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='relu', name=name + '_c1')(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='relu', name=name + '_c2')(x)
    return x


def _create_pm(od, ref):
    """Prediction module."""
    import keras
    from keras.regularizers import l2
    from model import ObjectDetector

    # スケール間で重みを共有するレイヤーの作成
    shared_layers = {}
    for pat_ix in range(len(od.pb_size_patterns)):
        prefix = 'pm-{}'.format(pat_ix)
        shared_layers[pat_ix] = {
            'c0': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv0'),
            'c1': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv1'),
            'c2': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv2'),
            'c3': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv3'),
            'conf': keras.layers.Conv2D(od.nb_classes, (1, 1), padding='same',
                                        kernel_regularizer=l2(1e-4),
                                        bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                                        bias_regularizer=l2(1e-4),  # bgの初期値が7.6とかなので、徐々に減らしたい
                                        activation='softmax',
                                        name=prefix + '_conf'),
            'loc': keras.layers.Conv2D(4, (1, 1), use_bias=False,  # 平均的には≒0のはずなのでバイアス無し
                                       kernel_regularizer=l2(1e-4),
                                       bias_regularizer=l2(1e-4),
                                       name=prefix + '_loc'),
        }

    def _pm(x, prefix, sl):
        # 非共有でsqueeze
        x = tk.dl.conv2d(64, (1, 1), activation='relu', name=prefix + '_sq')(x)
        # DenseBlock (BNだけ非共有)
        for branch in range(4):
            b = keras.layers.Dropout(0.25, name='{}_conv{}_drop'.format(prefix, branch))(x)
            b = sl['c' + str(branch)](b)
            b = keras.layers.BatchNormalization(name='{}_conv{}_bn'.format(prefix, branch))(b)
            b = keras.layers.Activation('relu', name='{}_conv{}_act'.format(prefix, branch))(b)
            x = keras.layers.Concatenate(name='{}_conv{}_concat'.format(prefix, branch))([x, b])
        # conf/loc
        conf = sl['conf'](x)
        loc = sl['loc'](x)
        conf = keras.layers.Reshape((-1, od.nb_classes), name=prefix + '_reshape_conf')(conf)
        loc = keras.layers.Reshape((-1, 4), name=prefix + '_reshape_loc')(loc)
        return conf, loc

    confs, locs = [], []
    for fm_count in ObjectDetector.FM_COUNTS:
        x = ref['up{}'.format(fm_count)]
        for pat_ix in range(len(od.pb_size_patterns)):
            prefix = 'pm{}-{}'.format(fm_count, pat_ix)
            conf, loc = _pm(x, prefix, shared_layers[pat_ix])
            confs.append(conf)
            locs.append(loc)
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)
    return confs, locs
