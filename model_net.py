"""ObjectDetectionのモデル。"""

import pytoolkit as tk
from model import ObjectDetector


def create_network(od: ObjectDetector):
    """モデルの作成。"""
    import keras
    import keras.backend as K
    from keras.regularizers import l2

    def _block(x, name):
        assert K.int_shape(x)[-1] <= 128
        in_filters = K.int_shape(x)[-1]
        x = tk.dl.conv2d(in_filters * 1, (3, 3), padding='same', activation='elu', name=name + '_c1')(x)
        x = tk.dl.conv2d(in_filters * 2, (3, 3), padding='same', activation='elu', name=name + '_c2')(x)
        return x

    def _branch(x, filters, name):
        x = tk.dl.conv2d(filters * 4, (1, 1), padding='same', activation='elu', name=name + '_c1')(x)
        x = tk.dl.conv2d(filters, (3, 3), padding='same', activation='elu', name=name + '_c2')(x)
        return x

    def _denseblock(x, name):
        assert K.int_shape(x)[-1] == 256
        for branch in range(4):
            b = _branch(x, 256 // 4, name=name + '_b' + str(branch))
            x = keras.layers.Concatenate()([x, b])
        x = tk.dl.conv2d(256, (1, 1), padding='same', activation='elu', name=name + '_sq')(x)
        return x

    def _downblock(x, name):
        assert K.int_shape(x)[-1] == 256
        x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', name=name + '_c1')(x)
        x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', name=name + '_c2')(x)
        return x

    def _ds(x, kernel_size, strides, name):
        x = keras.layers.AveragePooling2D(kernel_size, strides=strides, name=name)(x)
        return x

    def _us(x, name):
        assert K.int_shape(x)[-1] == 512
        x = tk.dl.conv2d(256, (1, 1), padding='same', activation='elu', name=name + '_c1')(x)
        x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', name=name + '_c2')(x)
        x = keras.layers.UpSampling2D(name=name + '_up')(x)
        return x

    def _upblock(x, b, name):
        assert K.int_shape(x)[-1] == 256
        assert K.int_shape(b)[-1] == 256
        x = keras.layers.Concatenate(name=name + '_concat')([x, b])
        return x

    # ベースネットワーク
    x = inputs = keras.layers.Input(od.input_size + (3,))
    if od.input_size == (645, 645):
        x = tk.dl.conv2d(32, (7, 7), strides=(2, 2), padding='valid', activation='elu', name='stage1_conv1')(x)
    else:
        x = tk.dl.conv2d(32, (3, 3), padding='same', activation='elu', name='stage1_conv1')(x)
    assert K.int_shape(x)[1] == 320
    x = tk.dl.conv2d(32, (3, 3), padding='same', activation='elu', name='stage1_conv2')(x)
    x = keras.layers.MaxPooling2D(name='stage1_ds')(x)
    assert K.int_shape(x)[1] == 160
    x = _block(x, 'stage2_block')
    x = _ds(x, (2, 2), (2, 2), 'stage2_ds')
    assert K.int_shape(x)[1] == 80
    x = _block(x, 'stage3_block')
    x = _ds(x, (2, 2), (2, 2), 'stage3_ds')
    assert K.int_shape(x)[1] == ObjectDetector.FM_COUNTS[0]
    x = _block(x, 'stage4_block1')
    x = _denseblock(x, 'stage4_block2')
    x = _denseblock(x, 'stage4_block3')
    assert K.int_shape(x)[-1] == 256

    net = {}
    net['out{}'.format(ObjectDetector.FM_COUNTS[0])] = x

    # downsampling
    for fm_count in ObjectDetector.FM_COUNTS[1:]:
        x = _downblock(x, 'out{}_downblock'.format(fm_count))
        x = _ds(x, (2, 2), (2, 2), 'out{}_ds'.format(fm_count))
        net['out{}'.format(fm_count)] = x

    # center
    x = _downblock(x, 'center_block')

    # upsampling
    for i, fm_count in enumerate(ObjectDetector.FM_COUNTS[::-1]):
        if i != 0:
            x = _us(x, 'out{}_us'.format(fm_count))
        x = _upblock(x, net['out{}'.format(fm_count)], 'out{}_upblock'.format(fm_count))
        net['out{}'.format(fm_count)] = x

    # prediction module

    def _conf(x, fm_count, size_ix, ar_ix):
        filters = od.nb_classes
        x = tk.dl.conv2d(filters * 2, (1, 1), padding='same', activation='elu',
                         name='pm{}-{}-{}-conf_sq'.format(fm_count, size_ix, ar_ix))(x)
        for depth in range(size_ix):
            x = tk.dl.conv2d(filters * 2, (3, 3), padding='same', activation='elu',
                             name='pm{}-{}-{}-conf_conv{}'.format(fm_count, size_ix, ar_ix, depth))(x)
        x = keras.layers.Conv2D(
            filters, (3, 3), padding='same',
            bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
            bias_regularizer=l2(1e-4),  # bgの初期値が7.6とかなので、徐々に減らしたい
            activation='softmax',
            name='pm{}-{}-{}-conf'.format(fm_count, size_ix, ar_ix))(x)
        x = keras.layers.Reshape((fm_count ** 2, od.nb_classes), name='pm{}-{}-{}-conf_reshape'.format(fm_count, size_ix, ar_ix))(x)
        # x = keras.layers.Activation('softmax', name='pm{}-{}-{}-conf_softmax'.format(fm_count, size_ix, ar_ix))(x)
        return x

    def _loc(x, fm_count, size_ix, ar_ix):
        filters = 4
        x = tk.dl.conv2d(x, filters * 2, (1, 1), padding='same', activation='elu',
                         name='pm{}-{}-{}-loc_sq'.format(fm_count, size_ix, ar_ix))(x)
        for depth in range(size_ix):
            x = tk.dl.conv2d(filters * 2, (3, 3), padding='same', activation='elu',
                             name='pm{}-{}-{}-loc_conv{}'.format(fm_count, size_ix, ar_ix, depth))(x)
        x = keras.layers.Conv2D(filters, (3, 3), padding='same',
                                name='pm{}-{}-{}-loc'.format(fm_count, size_ix, ar_ix))(x)
        x = keras.layers.Reshape((fm_count ** 2, 4), name='pm{}-{}-{}-loc_reshape'.format(fm_count, size_ix, ar_ix))(x)
        return x

    confs, locs = [], []
    for fm_count in ObjectDetector.FM_COUNTS:
        for size_ix in range(len(od.pb_size_ratios)):
            for ar_ix in range(len(od.aspect_ratios)):
                x = net['out{}'.format(fm_count)]
                confs.append(_conf(x, fm_count, size_ix, ar_ix))
                locs.append(_loc(x, fm_count, size_ix, ar_ix))
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs])

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # model.compile('nadam', od.loss, [od.loss_loc, od.acc_bg, od.acc_obj])
    model.compile(keras.optimizers.SGD(momentum=0.9, nesterov=True), od.loss, [od.loss_loc, od.acc_bg, od.acc_obj])
    return model
