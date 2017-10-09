"""ObjectDetectionのモデル。"""

import pytoolkit as tk

# ベースネットワークの種類: custom, resnet50, xception
_BASENET_TYPE = 'resnet50'


def create_pretrain_network(input_shape, nb_classes):
    """モデルの作成。"""
    import keras

    # downsampling (ベースネットワーク)
    x = inputs = keras.layers.Input(input_shape)
    x, _ = _create_basenet(x, freeze=False)
    # center
    x = _centerblock(x)

    # prediction
    x = keras.layers.GlobalAveragePooling2D(name='pool')(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer='l2', name='pretrain_predictions')(x)

    return keras.models.Model(inputs=inputs, outputs=x)


def create_network(od, freeze):
    """モデルの作成。"""
    import keras
    from model import ObjectDetector

    # downsampling (ベースネットワーク)
    x = inputs = keras.layers.Input(od.input_size + (3,))
    x, ref = _create_basenet(x, freeze)
    # center
    x = _centerblock(x)
    # auxnet
    _create_auxnet(inputs, ref)
    # upsampling
    for fm_count in ObjectDetector.FM_COUNTS[::-1]:
        x = _upblock(x, ref, fm_count)

    # prediction module
    outputs = _create_pm(od, ref)

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
        x = tk.dl.conv2d(32, (7, 7), strides=(2, 2), padding='valid', activation='elu', kernel_initializer='he_uniform', name='stage1_conv1')(x)  # 160x160
        x = tk.dl.conv2d(64, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='stage1_conv2')(x)
        x = keras.layers.MaxPooling2D(name='stage1_ds')(x)  # 80x80
        x = _denseblock(x, 64, 3, bottleneck=False, compress=False, name='stage2_block')
        for fm_count in ObjectDetector.FM_COUNTS:
            x = _downblock(x, ref, fm_count)
    elif _BASENET_TYPE == 'resnet50':
        basenet = keras.applications.ResNet50(include_top=False, input_tensor=x)
        if freeze:
            for layer in basenet.layers:
                layer.trainable = False
        ref['down{}'.format(40)] = basenet.get_layer(name='res4a_branch2a').input
        ref['down{}'.format(20)] = basenet.get_layer(name='res5a_branch2a').input
        ref['down{}'.format(10)] = basenet.get_layer(name='avg_pool').input
        x = ref['down{}'.format(10)]
        for fm_count in (5,):
            x = _downblock(x, ref, fm_count)
    elif _BASENET_TYPE == 'xception':
        basenet = keras.applications.Xception(include_top=False, input_tensor=x)
        if freeze:
            for layer in basenet.layers:
                layer.trainable = False
        ref['down{}'.format(40)] = basenet.get_layer(name='block4_sepconv1_act').input
        ref['down{}'.format(20)] = basenet.get_layer(name='block13_sepconv1_act').input
        ref['down{}'.format(10)] = basenet.get_layer(name='block14_sepconv2_act').output
        x = ref['down{}'.format(10)]
        for fm_count in (5,):
            x = _downblock(x, ref, fm_count)
    else:
        assert False

    for fm_count in ObjectDetector.FM_COUNTS:
        assert 'down{}'.format(fm_count) in ref
        x = ref['down{}'.format(fm_count)]
        assert K.int_shape(x)[1] == fm_count

    return x, ref


def _create_auxnet(x, ref):
    """補助的なネットワークの作成。"""
    import keras
    import keras.backend as K
    from model import ObjectDetector

    if _BASENET_TYPE == 'resnet50':
        shared_layers = {
            'c1': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_conv1'),
            'c2': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_conv2'),
            'c3': keras.layers.Conv2D(64, (1, 1), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_conv3'),
        }

        def _shared_block(x):
            b = shared_layers['c1'](x)
            x = keras.layers.Concatenate()([x, b])
            b = shared_layers['c2'](x)
            x = keras.layers.Concatenate()([x, b])
            x = shared_layers['c3'](x)
            return x

        x = keras.layers.AveragePooling2D((3, 3), (2, 2), padding='same', name='aux_stage0_ds')(x)
        x = tk.dl.conv2d(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_stage1_c')(x)
        x = keras.layers.AveragePooling2D((3, 3), (2, 2), padding='valid', name='aux_stage1_ds')(x)
        x = tk.dl.conv2d(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_stage2_c1')(x)
        x = tk.dl.conv2d(64, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='aux_stage2_c2')(x)
        x = _shared_block(x)
        for fm_count in ObjectDetector.FM_COUNTS:
            x = keras.layers.AveragePooling2D((3, 3), (2, 2), padding='same', name='aux{}_ds'.format(fm_count))(x)
            x = _shared_block(x)
            assert K.int_shape(x)[1] == fm_count
            ref['aux{}'.format(fm_count)] = x
    else:
        assert False


def _denseblock(x, inc_filters, branches, bottleneck, compress, name):
    import keras
    import keras.backend as K

    for branch in range(branches):
        if bottleneck:
            b = tk.dl.conv2d(inc_filters * 4, (1, 1), padding='same', activation='elu', kernel_initializer='he_uniform', name=name + '_b' + str(branch) + '_c1')(x)
            b = keras.layers.Dropout(0.25)(b)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=name + '_b' + str(branch) + '_c2')(b)
        else:
            b = keras.layers.Dropout(0.25)(x)
            b = tk.dl.conv2d(inc_filters * 1, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=name + '_b' + str(branch))(b)
        x = keras.layers.Concatenate(name=name + '_b' + str(branch) + '_cat')([x, b])

    if compress:
        x = tk.dl.conv2d(K.int_shape(x)[-1] // 2, (1, 1), padding='same', activation='elu', kernel_initializer='he_uniform', name=name + '_sq')(x)

    return x


def _downblock(x, ref, fm_count):
    import keras
    import keras.backend as K

    if K.int_shape(x)[-1] > 256:
        x = tk.dl.conv2d(256, (1, 1), activation='elu', kernel_initializer='he_uniform', name='down{}_sq'.format(fm_count))(x)
    assert K.int_shape(x)[-1] == 256

    x = keras.layers.MaxPooling2D(name='down{}_ds'.format(fm_count))(x)
    assert K.int_shape(x)[1] == fm_count

    x = _denseblock(x, 64, 8, bottleneck=True, compress=True, name='down{}_block'.format(fm_count))

    ref['down{}'.format(fm_count)] = x
    return x


def _centerblock(x):

    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='center_c1')(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='center_c2')(x)

    return x


def _upblock(x, ref, fm_count):
    import keras
    import keras.backend as K

    t = ref['down{}'.format(fm_count)]
    a = ref['aux{}'.format(fm_count)]

    if K.int_shape(x)[1] != fm_count:
        x = keras.layers.UpSampling2D(name='up{}_us'.format(fm_count))(x)
    assert K.int_shape(x)[1] == fm_count

    x = tk.dl.conv2d(256, (3, 3), padding='same', activation=None, kernel_initializer='he_uniform', name='up{}_c1'.format(fm_count))(x)
    t = tk.dl.conv2d(256, (1, 1), padding='same', activation=None, kernel_initializer='he_uniform', name='up{}_lt'.format(fm_count))(t)
    a = tk.dl.conv2d(256, (1, 1), padding='same', activation=None, kernel_initializer='he_uniform', name='up{}_ax'.format(fm_count))(a)
    x = keras.layers.Add(name='up{}_mix'.format(fm_count))([x, t, a])
    x = keras.layers.BatchNormalization(name='up{}_mix_bn'.format(fm_count))(x)
    x = keras.layers.Activation(activation='elu', name='up{}_mix_act'.format(fm_count))(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='up{}_c2'.format(fm_count))(x)
    x = tk.dl.conv2d(256, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name='up{}_c3'.format(fm_count))(x)

    ref['up{}'.format(fm_count)] = x
    return x


def _create_pm(od, ref):
    """Prediction module."""
    import keras
    from keras.regularizers import l2
    from model import ObjectDetector

    # スケール間で重みを共有するレイヤー
    shared_layers = {}
    for pat_ix in range(len(od.pb_size_patterns)):
        prefix = 'pm-{}'.format(pat_ix)
        shared_layers[pat_ix] = {
            'sq': keras.layers.Conv2D(64, (1, 1), padding='same', activation='elu', kernel_initializer='he_uniform', name=prefix + '_sq'),
            'c0': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=prefix + '_c0'),
            'c1': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=prefix + '_c1'),
            'c2': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=prefix + '_c2'),
            'c3': keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu', kernel_initializer='he_uniform', name=prefix + '_c3'),
            'conf': keras.layers.Conv2D(od.nb_classes, (1, 1), padding='same',
                                        kernel_initializer='zeros',
                                        kernel_regularizer=l2(1e-4),
                                        bias_initializer=tk.dl.od_bias_initializer(od.nb_classes),
                                        bias_regularizer=l2(1e-4),  # bgの初期値が7.6とかなので、徐々に減らしたい
                                        activation='softmax',
                                        name=prefix + '_conf'),
            'loc': keras.layers.Conv2D(4, (1, 1), padding='same', use_bias=False,  # 平均的には≒0のはずなのでバイアス無し
                                       kernel_initializer='zeros',
                                       kernel_regularizer=l2(1e-4),
                                       name=prefix + '_loc'),
            'iou': keras.layers.Conv2D(1, (3, 3), padding='same',
                                       kernel_initializer='zeros',
                                       kernel_regularizer=l2(1e-4),
                                       activation='sigmoid',
                                       name=prefix + '_iou'),
        }

    def _pm(x, prefix, shlayers):
        # squeeze
        x = shlayers['sq'](x)
        # DenseBlock (BNだけ非共有)
        for branch in range(4):
            b = keras.layers.Dropout(0.25, name='{}_c{}_drop'.format(prefix, branch))(x)
            b = shlayers['c' + str(branch)](b)
            x = keras.layers.Concatenate(name='{}_c{}_cat'.format(prefix, branch))([x, b])
        # conf/loc/iou
        conf = shlayers['conf'](x)
        loc = shlayers['loc'](x)
        iou = shlayers['iou'](keras.layers.Concatenate()([x, loc]))
        # reshape
        conf = keras.layers.Reshape((-1, od.nb_classes), name=prefix + '_reshape_conf')(conf)
        loc = keras.layers.Reshape((-1, 4), name=prefix + '_reshape_loc')(loc)
        iou = keras.layers.Reshape((-1, 1), name=prefix + '_reshape_iou')(iou)
        return conf, loc, iou

    confs, locs, ious = [], [], []
    for fm_count in ObjectDetector.FM_COUNTS:
        x = ref['up{}'.format(fm_count)]
        for pat_ix in range(len(od.pb_size_patterns)):
            prefix = 'pm{}-{}'.format(fm_count, pat_ix)
            conf, loc, iou = _pm(x, prefix, shared_layers[pat_ix])
            confs.append(conf)
            locs.append(loc)
            ious.append(iou)
    confs = keras.layers.Concatenate(axis=-2, name='output_confs')(confs)
    locs = keras.layers.Concatenate(axis=-2, name='output_locs')(locs)
    ious = keras.layers.Concatenate(axis=-2, name='output_ious')(ious)

    # いったんくっつける (損失関数の中で分割して使う)
    outputs = keras.layers.Concatenate(axis=-1, name='outputs')([confs, locs, ious])

    return outputs


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

    objclasses = keras.layers.Lambda(lambda c: K.argmax(c[:, :, 1:], axis=-1) + 1, K.int_shape(confs)[1:-1])(confs)
    objconfs = keras.layers.Lambda(lambda x: K.max(x[0][:, :, 1:], axis=-1) * x[1][:, :, 0], K.int_shape(confs)[1:-1])([confs, ious])
    locs = keras.layers.Lambda(lambda l: K.clip(l * od.pb_scales + od.pb_locs, 0, 1), K.int_shape(locs))(locs)

    return keras.models.Model(inputs=model.inputs, outputs=[objclasses, objconfs, locs])
