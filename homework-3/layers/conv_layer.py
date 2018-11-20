# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

batch_size = 100


class ConvLayer():
    """
    2D convolutional layer.
    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs.
    Arguments:
        inputs: Integer, the channels number of input.
        filters: Integer, the number of filters in the convolution.
        kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
        pad: Integer, the size of padding area.
        trainable: Boolean, whether this layer is trainable.
    """

    def __init__(self, inputs,
                 filters,
                 kernel_size,
                 pad,
                 trainable=True):
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.pad = pad
        assert pad < kernel_size, "pad should be less than kernel_size"
        self.trainable = trainable
        self.XavierInit()

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def XavierInit(self):
        raw_std = (2 / (self.inputs + self.filters)) ** 0.5
        init_std = raw_std * (2 ** 0.5)

        self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
        self.b = np.random.normal(0, init_std, (self.filters,))

    def forward(self, Input):
        '''
        forward method: perform convolution operation on the input.
        Agrs:
            Input: A batch of images, shape=(batch_size, channels, height, width)
        '''
        out = None
        self.Input = Input
        N, C, H, W = Input.shape
        F, _, HH, WW = self.W.shape
        P = self.pad
        Ho = 1 + (H + 2 * P - HH)
        Wo = 1 + (W + 2 * P - WW)
        x_pad = np.zeros((N, C, H + 2 * P, W + 2 * P))
        x_pad[:, :, P:P + H, P:P + W] = self.Input
        out = np.zeros((N, F, Ho, Wo))
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    out[:, f, i, j] = np.sum(x_pad[:, :, i : i  + HH, j : j  + WW] * self.W[f, :, :, :],axis=(1, 2, 3))

            out[:, f, :, :] += self.b[f]
        return out


    def backward(self, delta):

        N, F, H1, W1 = delta.shape
        x = self.Input
        w = self.W
        b = self.b

        N, C, H, W = x.shape
        HH = w.shape[2]
        WW = w.shape[3]
        P = self.pad

        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        x_pad = np.pad(x, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')
        dx_pad = np.pad(dx, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')
        db = np.sum(delta, axis=(0, 2, 3))

        for n in range(N):
            for i in range(H1):
                for j in range(W1):
                    # Window we want to apply the respective f th filter over (C, HH, WW)
                    x_window = x_pad[n, :, i : i  + HH, j : j  + WW]
                    for f in range(F):
                        dw[f] += x_window * delta[n, f, i, j]  # F,C,HH,WW
                        # C,HH,WW
                        dx_pad[n, :, i : i  + HH, j : j  + WW] += w[f] * delta[n, f, i, j]

        dx = dx_pad[:, :, P:P + H, P:P + W]
        self.grad_W = dw
        self.grad_b = db
        return dx
