"""ObjectDetectionのモデル。"""

import pytoolkit as tk
from model import ObjectDetector

INPUT_SIZE = (325, 325)


def create_pretrain_network(input_shape, nb_classes):
    """モデルの作成。"""
    import keras
    import keras.backend as K

    # ベースネットワーク
    x = inputs = keras.layers.Input(input_shape)
    x = create_basenet(x)
    assert K.int_shape(x)[1] == input_shape[0] // 4

    # downsampling (最初の部分のみ)
    fm_count = ObjectDetector.FM_COUNTS[0]
    x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
    x = _downblock(x, 'down{}_block'.format(fm_count))

    # prediction
    x = keras.layers.GlobalMaxPooling2D('avgpool')(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2', name='pretrain_predictions')(x)

    return keras.models.Model(inputs=inputs, outputs=x)


def create_network(od: ObjectDetector):
    """モデルの作成。"""
    import keras
    import keras.backend as K

    # ベースネットワーク
    x = inputs = keras.layers.Input(od.input_size + (3,))
    x = create_basenet(x)
    assert K.int_shape(x)[1] == ObjectDetector.FM_COUNTS[0] * 2

    # downsampling
    ref = {}
    for fm_count in ObjectDetector.FM_COUNTS:
        x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
        x = _downblock(x, 'down{}_block'.format(fm_count))
        assert K.int_shape(x)[1] == fm_count
        ref['down{}'.format(fm_count)] = x

    # center
    x = _downblock(x, 'center_block')

    # upsampling
    for fm_count in ObjectDetector.FM_COUNTS[::-1]:
        if fm_count != ObjectDetector.FM_COUNTS[-1]:
            x = keras.layers.UpSampling2D(name='up{}_us'.format(fm_count))(x)
        x = _upblock(x, ref['down{}'.format(fm_count)], 'up{}_block'.format(fm_count))
        ref['up{}'.format(fm_count)] = x

    # prediction module
    confs, locs = create_pm(od, ref)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    return keras.models.Model(inputs=inputs, outputs=outputs)


def create_basenet(x):
    """ベースネットワークの作成。"""
    import keras
    x = tk.dl.conv2d(32, (7, 7), strides=(2, 2), padding='valid', activation='relu', name='stage1_conv1')(x)  # 160x160
    x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='stage1_conv2')(x)
    x = keras.layers.MaxPooling2D(name='stage1_ds')(x)  # 80x80。DenseNetを参考に最初だけMax。
    x = _denseblock(x, 64, 3, bottleneck=False, compress=False, name='stage2_block')
    return x


def _denseblock(x, inc_filters, branches, bottleneck, compress, name):
    import keras
    import keras.backend as K
    for branch in range(branches):
        if bottleneck:
            b = tk.dl.conv2d(inc_filters * 4, (1, 1), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c1')(x)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c2')(b)
        else:
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='relu', name=name + '_b' + str(branch))(x)
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
    x = keras.layers.Concatenate(name=name + '_concat')([x, b])
    x = tk.dl.conv2d(256, (1, 1), padding='same', activation='relu', name=name + '_sq')(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='relu', name=name + '_c')(x)
    return x


def create_pm(od, ref):
    """Prediction module."""
    import keras
    from keras.regularizers import l2

    # スケール間で重みを共有するレイヤーの作成
    shared_layers = {}
    for size_ix in range(len(od.pb_size_ratios)):
        for ar_ix in range(len(od.aspect_ratios)):
            prefix = 'pm-{}-{}'.format(size_ix, ar_ix)
            shared_layers[(size_ix, ar_ix)] = {
                'c0': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv0'),
                'c1': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv1'),
                'c2': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv2'),
                'c3': keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name=prefix + '_conv3'),
                'conf': keras.layers.Conv2D(od.nb_classes, (1, 1), padding='same',
                                            bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                                            bias_regularizer=l2(1e-4),  # bgの初期値が7.6とかなので、徐々に減らしたい
                                            activation='softmax',
                                            name=prefix + '_conf'),
                'loc': keras.layers.Conv2D(4, (1, 1), use_bias=False,  # 平均的には≒0のはずなのでバイアス無し
                                           name=prefix + '_loc'),
            }

    def _pm(x, prefix, sl):
        # 非共有でsqueeze
        x = tk.dl.conv2d(64, (1, 1), activation='relu', name=prefix + '_sq')(x)
        # DenseBlock (BNだけ非共有)
        for branch in range(4):
            b = sl['c' + str(branch)](x)
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
        for size_ix in range(len(od.pb_size_ratios)):
            for ar_ix in range(len(od.aspect_ratios)):
                prefix = 'pm{}-{}-{}'.format(fm_count, size_ix, ar_ix)
                conf, loc = _pm(x, prefix, shared_layers[(size_ix, ar_ix)])
                confs.append(conf)
                locs.append(loc)
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)
    return confs, locs
