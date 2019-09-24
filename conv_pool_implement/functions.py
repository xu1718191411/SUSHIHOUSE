import numpy as np


def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def softmax(x):

    if x.ndim == 1:
        max = np.max(x, axis=1)
        x = x - max
        result = np.exp(x) / np.sum(np.exp(x))
        return result
    else:
        x = x.T
        max = np.max(x,axis=0)
        x = x - max
        result = np.exp(x) / np.sum(np.exp(x),axis=0)
        result = result.T
        return result

def cross_entropy_error(x,t):
    delta = 1e-7
    return np.sum(-1 * t * np.log(x + delta))/ x.shape[0]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col