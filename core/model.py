# Author: Jerry Xia <jerry_xiazj@outlook.com>

import config as CFG
import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Identity")

    def call(self, input):
        return input


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSigmoid")
        self.relu6 = tf.keras.layers.ReLU(max_value=6)

    def call(self, input):
        # return self.relu6(input + np.float32(3.0)) * np.float32(1.0 / 6.0)
        return self.relu6(input + 3.0) / 6.0


class HardSwish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSwish")
        self.hardSigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hardSigmoid(input)


class Squeeze(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Squeeze")

    def call(self, input):
        x = tf.keras.backend.squeeze(input, 1)
        x = tf.keras.backend.squeeze(x, 1)
        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        bn: bool,
        nl: str
    ):
        super().__init__(name="ConvBlock")
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=stride,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            use_bias=bias
        )
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.99) if bn else Identity()
        _available_act = {
            "relu": tf.keras.layers.ReLU(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax()
        }
        self.act = _available_act[nl] if nl else Identity()

    def call(self, input):
        x = self.conv(input)
        x = self.norm(x)
        x = self.act(x)
        return x


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="SqueezeExcite")

    def build(self, input_shape):
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=tuple(map(int, input_shape[1:3]))
        )
        self.conv1 = ConvBlock(
            filters=int(input_shape[3]) // 4, kernel_size=1, stride=1,
            bias=False, bn=False, nl="relu"
        )
        self.conv2 = ConvBlock(
            filters=int(input_shape[3]), kernel_size=1, stride=1,
            bias=False, bn=False, nl="hsigmoid"
        )
        super().build(input_shape)

    def call(self, input):
        x = self.pool(input)
        x = self.conv1(x)
        x = self.conv2(x)
        return input * x


class Bneck(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channel: int,
        exp_channel: int,
        kernel_size: int,
        stride: int,
        se: bool,
        nl: str
    ):
        self.stride = stride
        self.out_channel = out_channel
        super().__init__(name="BottleNeck")
        self.expand = ConvBlock(
            filters=exp_channel, kernel_size=1, stride=1,
            bias=False, bn=True, nl=nl
        )
        self.dwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=stride,
            padding="same", use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(1e-5),
        )
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.se = SqueezeExcite() if se else Identity()
        _available_act = {
            "relu": tf.keras.layers.ReLU(),
            "hswish": HardSwish()
        }
        self.act = _available_act[nl] if nl else Identity()
        self.project = ConvBlock(
            filters=out_channel, kernel_size=1, stride=1,
            bias=False, bn=True, nl=None
        )

    def build(self, input_shape):
        self.in_channel = int(input_shape[3])
        self.connect = self.stride == 1 and self.in_channel == self.out_channel
        super().build(input_shape)

    def call(self, input):
        x = self.expand(input)
        x = self.dwise(x)
        x = self.norm(x)
        x = self.se(x)
        x = self.act(x)
        x = self.project(x)
        return input + x if self.connect else x


class LastStage(tf.keras.layers.Layer):
    def __init__(
        self,
        first_channel: int,
        second_channel: int,
        num_classes: int
    ):
        super().__init__(name="LastStage")
        self.conv1 = ConvBlock(
            filters=first_channel, kernel_size=1, stride=1,
            bias=False, bn=True, nl="hswish"
        )
        self.conv2 = ConvBlock(
            filters=second_channel, kernel_size=1, stride=1,
            bias=True, bn=False, nl="hswish"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.conv3 = ConvBlock(
            filters=num_classes, kernel_size=1, stride=1,
            bias=True, bn=False, nl="softmax"
        )
        self.squeeze = Squeeze()

    def build(self, input_shape):
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=tuple(map(int, input_shape[1:3]))
        )

    def call(self, input):
        x = self.conv1(input)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.squeeze(x)
        return x


class MobileNetv3_small(tf.keras.Model):
    def __init__(self, num_classes: int):
        super().__init__(name="MobileNetv3_small")

        self.first_stage = ConvBlock(
            filters=16, kernel_size=3, stride=2,
            bias=False, bn=True, nl="hswish"
        )

        self.middle_stage = tf.keras.Sequential()
        _bneck_config = [
            # k  exp   out   SE       NL       s
            [3,  16,   16,  True,   "relu",    2],
            [3,  72,   24,  False,  "relu",    2],
            [3,  88,   24,  False,  "relu",    1],
            [5,  96,   40,  True,   "hswish",  2],
            [5,  240,  40,  True,   "hswish",  1],
            [5,  240,  40,  True,   "hswish",  1],
            [5,  120,  48,  True,   "hswish",  1],
            [5,  144,  48,  True,   "hswish",  1],
            [5,  288,  96,  True,   "hswish",  2],
            [5,  576,  96,  True,   "hswish",  1],
            [5,  576,  96,  True,   "hswish",  1],
        ]
        for _k, _exp, _out, _se, _nl, _s in _bneck_config:
            self.middle_stage.add(
                Bneck(
                    out_channel=_out, exp_channel=_exp,
                    kernel_size=_k, stride=_s, se=_se, nl=_nl
                )
            )

        self.last_stage = LastStage(
            first_channel=576,
            second_channel=1024,
            num_classes=num_classes
        )

    def call(self, input):
        x = self.first_stage(input)
        x = self.middle_stage(x)
        x = self.last_stage(x)
        return x


if __name__ == "__main__":
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
    output_tensor = MobileNetv3_small(1001)(input_tensor)
    model = tf.keras.Model(
        inputs=input_tensor,
        outputs=output_tensor,
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=CFG.lr_init, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()
