"""ObjectDetectionのモデル。"""

import pytoolkit as tk
from model import ObjectDetector

_LARGE_IMAGE_SIZE = False
INPUT_SIZE = (639, 639) if _LARGE_IMAGE_SIZE else (320, 320)


def create_network(od: ObjectDetector):
    """モデルの作成。"""
    import keras
    import keras.backend as K

    def _branch(x, filters, name):
        x = tk.dl.conv2d(filters, (3, 3), padding='same', activation='relu', name=name + '_c1')(x)
        x = keras.layers.Dropout(0.25, name=name + '_drop')(x)
        x = tk.dl.conv2d(filters, (3, 3), padding='same', activation='relu', name=name + '_c2')(x)
        return x

    def _downblock(input_x, name):
        assert K.int_shape(input_x)[-1] in (128, 256)
        x = inputs = keras.layers.Input(K.int_shape(input_x)[1:])
        for branch in range(4):
            b = _branch(x, 256 // 4, name=name + '_b' + str(branch))
            x = keras.layers.Concatenate()([x, b])
        x = tk.dl.conv2d(256, (1, 1), padding='same', activation='relu', name=name + '_comp')(x)
        return keras.engine.topology.Container(inputs, x, name)(input_x)

    def _us(x, name):
        assert K.int_shape(x)[-1] == 256
        x = tk.dl.conv2d(256, (3, 3), padding='same', activation='relu', name=name + '_c')(x)
        x = keras.layers.UpSampling2D(name=name + '_up')(x)
        return x

    def _upblock(x, b, name):
        assert K.int_shape(x)[-1] == 256
        assert K.int_shape(b)[-1] == 256
        x = keras.layers.Concatenate(name=name + '_concat')([x, b])
        x = tk.dl.conv2d(256, (3, 3), padding='same', activation='relu', name=name + '_c')(x)
        return x

    # ベースネットワーク
    x = inputs = keras.layers.Input(od.input_size + (3,))
    x = create_basenet(_LARGE_IMAGE_SIZE)(x)
    assert K.int_shape(x)[1] == ObjectDetector.FM_COUNTS[0] * 2

    # downsampling
    ref = {}
    for i, fm_count in enumerate(ObjectDetector.FM_COUNTS):
        x = keras.layers.AveragePooling2D(name='down{}_ds'.format(fm_count))(x)
        x = _downblock(x, 'down{}_block1'.format(fm_count))  # for lateral connection
        assert K.int_shape(x)[1] == fm_count
        ref['down{}'.format(fm_count)] = x
        x = _downblock(x, 'down{}_block2'.format(fm_count))  # for down sampling

    # # center
    # x = _downblock(x, 'center_block')

    # upsampling
    for i, fm_count in enumerate(ObjectDetector.FM_COUNTS[::-1]):
        if i != 0:
            x = _us(x, 'up{}_us'.format(fm_count))
        x = _upblock(x, ref['down{}'.format(fm_count)], 'up{}_block'.format(fm_count))
        ref['up{}'.format(fm_count)] = x

    # prediction module
    pm, pm_size = create_pm(od)
    conf_reshape = keras.layers.Reshape((-1, od.nb_classes), name='pm-conf_reshape')
    loc_reshape = keras.layers.Reshape((-1, 4), name='pm-loc_reshape')
    confs, locs = [], []
    for fm_count in ObjectDetector.FM_COUNTS:
        x = ref['up{}'.format(fm_count)]
        pm_output = pm(x)
        assert len(pm_output) == pm_size * 2
        conf, loc = pm_output[:pm_size], pm_output[pm_size:]
        confs.extend([conf_reshape(x) for x in conf])
        locs.extend([loc_reshape(x) for x in loc])
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    return keras.models.Model(inputs=inputs, outputs=outputs)


def create_basenet(large_size):
    """ベースネットワークの作成。"""
    import keras
    import keras.backend as K

    def _denseblock(x, name):
        assert K.int_shape(x)[-1] >= 64
        for branch in range(4):
            b = tk.dl.conv2d(256 // 4, (1, 1), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c1')(x)
            b = tk.dl.conv2d(256 // 1, (3, 3), padding='same', activation='relu', name=name + '_b' + str(branch) + '_c2')(b)
            x = keras.layers.Concatenate()([x, b])
        x = tk.dl.conv2d(256, (1, 1), padding='same', activation='relu', name=name + '_sq')(x)
        return x

    x = inputs = keras.layers.Input((None, None, 3))
    if large_size:
        x = tk.dl.conv2d(16, (7, 7), strides=(2, 2), padding='same', activation='relu', name='stage1_conv1_l')(x)
    else:
        x = tk.dl.conv2d(16, (3, 3), padding='same', activation='relu', name='stage1_conv1')(x)
    x = tk.dl.conv2d(32, (3, 3), padding='same', activation='relu', name='stage1_conv2')(x)
    x = keras.layers.AveragePooling2D(name='stage1_ds')(x)
    x = tk.dl.conv2d(32, (3, 3), padding='same', activation='relu', name='stage2_conv1')(x)
    x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='stage2_conv2')(x)
    x = keras.layers.AveragePooling2D(name='stage2_ds')(x)
    x = tk.dl.conv2d(64, (3, 3), padding='same', activation='relu', name='stage3_conv1')(x)
    x = tk.dl.conv2d(128, (3, 3), padding='same', activation='relu', name='stage3_conv2')(x)
    return keras.engine.topology.Container(inputs, x, 'basenet')


def create_pm(od: ObjectDetector):
    """Prediction module."""
    import keras
    inputs = x = keras.layers.Input((None, None, 256))
    confs, locs = [], []
    for size_ix in range(len(od.pb_size_ratios)):
        for ar_ix in range(len(od.aspect_ratios)):
            conf, loc = _create_pm(x, size_ix, ar_ix, od)
            confs.append(conf)
            locs.append(loc)
    outputs = confs + locs
    return keras.engine.topology.Container(inputs, outputs, 'pm'), len(confs)


def _create_pm(x, size_ix, ar_ix, od: ObjectDetector):
    import keras
    from keras.regularizers import l2
    filters = od.nb_classes + 4
    # まずはチャンネル数削減
    x = tk.dl.conv2d(filters * 2, (1, 1), padding='same', activation='relu', use_bn=False,
                     name='pm-{}-{}_sq'.format(size_ix, ar_ix))(x)
    # 軽くDenseNet
    for depth in range(2):
        b = tk.dl.conv2d(filters, (3, 3), padding='same', activation='relu', use_bn=False,
                         name='pm-{}-{}_conv{}'.format(size_ix, ar_ix, depth))(x)
        x = keras.layers.Concatenate()([x, b])
    # 最終層(conf)
    conf = tk.dl.conv2d(od.nb_classes, (3, 3), padding='same',
                        bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                        bias_regularizer=l2(1e-4),  # bgの初期値が7.6とかなので、徐々に減らしたい
                        activation='softmax', use_bn=False,
                        name='pm-{}-{}-conf'.format(size_ix, ar_ix))(x)
    # 最終層(loc)
    loc = tk.dl.conv2d(4, (3, 3), padding='same', activation=None, use_bn=False,
                       name='pm-{}-{}-loc'.format(size_ix, ar_ix))(x)
    return conf, loc
