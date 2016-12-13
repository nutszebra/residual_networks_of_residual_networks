import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class BN_ReLU_Conv_BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_sizes=(3, 3), strides=(1, 1), pads=(1, 1), probability=1.0):
        super(BN_ReLU_Conv_BN_ReLU_Conv, self).__init__()
        modules = []
        modules += [('bn1', L.BatchNormalization(in_channel))]
        modules += [('conv1', L.Convolution2D(in_channel, out_channel, filter_sizes[0], strides[0], pads[0]))]
        modules += [('bn2', L.BatchNormalization(out_channel))]
        modules += [('conv2', L.Convolution2D(out_channel, out_channel, filter_sizes[1], strides[1], pads[1]))]
        if not in_channel == out_channel:
            stride = int(np.max(strides))
            modules += [('projection', BN_ReLU_Conv(in_channel, out_channel, 1, stride, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.pads = pads
        self.probability = probability

    def _conv_initialization(self, conv):
        conv.W.data = self.weight_relu_initialization(conv)
        conv.b.data = self.bias_initialization(conv, constant=0)

    def weight_initialization(self):
        self._conv_initialization(self.conv1)
        self._conv_initialization(self.conv2)
        if not self.in_channel == self.out_channel:
            self['projection'].weight_initialization()

    @staticmethod
    def _count_conv(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        count = self._count_conv(self.conv1) + self._count_conv(self.conv2)
        if self.in_channel == self.out_channel:
            return count
        else:
            return count + self['projection'].count_parameters()

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.strides:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        if train is True and self.probability <= np.random.rand():
            # do nothing
            return x
        else:
            batch, channel, height, width = x.data.shape
            _, in_channel, _, _ = self.conv1.W.data.shape
            # if block to execute downsampling is dropped, in_channel is different
            x = self.concatenate_zero_pad(x, (batch, in_channel, height, width), x.volatile, type(x.data))
            h = self.bn1(x, test=not train)
            h = F.relu(h)
            h = self.conv1(h)
            h = self.bn2(h, test=not train)
            h = F.relu(h)
            h = self.conv2(h)
            # expectation
            if train is False:
                h = h * self.probability
            if self.in_channel == self.out_channel:
                # not downsampling, so zero pad
                h = h + self.concatenate_zero_pad(self.maybe_pooling(x), h.data.shape, h.volatile, type(h.data))
            else:
                # downsampling
                h = h + self['projection'](x, train=train)
            return h


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, n=13, stride_at_first_layer=2, probability=(1.0, ) * 13):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.out_channel = out_channel
        self.n = n
        self.stride_at_first_layer = stride_at_first_layer
        self.probability = probability
        super(ResBlock, self).__init__()
        modules = []
        modules += [('projection', BN_ReLU_Conv(in_channel, out_channel, 1, stride_at_first_layer, 0))]
        for i in six.moves.range(n):
            modules += [('block{}'.format(i), BN_ReLU_Conv_BN_ReLU_Conv(in_channel, out_channel, (3, 3), (stride_at_first_layer, 1), (1, 1), probability[i]))]
            stride_at_first_layer = 1
            in_channel = out_channel
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = x
        for i in six.moves.range(self.n):
            h = self['block{}'.format(i)](h, train)
        batch, ch, height, width = h.data.shape
        _, _, height_x, width_x = x.data.shape
        stride = (int(height_x / height), int(width_x / width))
        self['projection'].conv.stride = stride
        return h + self['projection'](x, train)


class ResnetOfResnet(nutszebra_chainer.Model):

    def __init__(self, category_num, N=(int(40 / 3 / 2),) * 3, out_channels=(16 * 2, 32 * 2, 64 * 2), p=(1.0, 0.5)):
        super(ResnetOfResnet, self).__init__()
        # conv
        modules = [('conv1', BN_ReLU_Conv(3, out_channels[0], 3, 1, 1))]
        # channels
        drop_probability = ResnetOfResnet.linear_schedule(p[0], p[1], N)
        in_channel = out_channels[0]
        strides = [1] + [2] * (len(N) - 1)
        for i in six.moves.range(len(N)):
            modules += [('res_block{}'.format(i), ResBlock(in_channel, out_channels[i], N[i], strides[i], drop_probability[i]))]
            in_channel = out_channels[i]
        modules += [('projection', BN_ReLU_Conv(out_channels[0], out_channels[-1], 1, 2 * (len(N) - 1), 0))]
        modules += [('linear', BN_ReLU_Conv(out_channels[-1], category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.N = N
        self.out_channels = out_channels
        self.p = p
        self.strides = strides
        self.drop_probability = drop_probability
        self.category_num = category_num
        self.name = 'resnet_of_resnet_{}_{}_{}_{}'.format(category_num, N, out_channels, p)

    @staticmethod
    def linear_schedule(bottom_layer, top_layer, N):
        total_block = sum(N)

        def y(x):
            return (float(-1 * bottom_layer) + top_layer) / (total_block) * x + bottom_layer
        theta = []
        count = 0
        for num in N:
            tmp = []
            for i in six.moves.range(count, count + num):
                tmp.append(y(i))
            theta.append(tmp)
            count += num
        return theta

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = self.conv1(x, train=train)
        keep_h = h
        for i in six.moves.range(len(self.N)):
            name = 'res_block{}'.format(i)
            h = self[name](h, train)
        # root projection
        batch, ch, height, width = h.data.shape
        _, _, height_x, width_x = keep_h.data.shape
        stride = (int(height_x / height), int(width_x / width))
        self['projection'].conv.stride = stride
        h = h + self['projection'](keep_h, train)
        # global average pooling
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(h, train=train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
