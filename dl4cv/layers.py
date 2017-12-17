import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################

    stride = conv_param['stride']
    padding = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    h_out = 1 + (H + 2 * padding - HH) // stride
    w_out = 1 + (W + 2 * padding - WW) // stride

    # apply padding to input
    if padding > 0:
        # Explanation of pad_with for the 2-dim case
        # [(0, 1), (0, 1)]
        #          ^^^^^^ ------ padding for second dimension
        # ^^^^^^ --------------  padding for first dimension
        #
        #  ^ ------------------  no padding at the beginning of the first axis
        #     ^ ---------------  pad with one "value" at the end of the first axis.
        pad_with = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        x = np.pad(x, pad_with, 'constant', constant_values=0)

    print('Padding: ', padding)
    print('Stride: ', stride)
    print('Input size: ', x.shape)
    print('Filter size: ', w.shape)
    print('output dimensions: ', (N, F, h_out, w_out))

    # out: Output data, of shape (N, F, H', W')
    out = np.zeros((N, F, h_out, w_out))
    # convolve over the image with the naive (inefficient) way

    # for each sample in X
    for sample_idx in range(N):
        sample = x[sample_idx]

        # for each filter do
        for filter_idx in range(F):
            filter = w[filter_idx]

            # iterate over each pixel in the input image
            out_x = 0
            for kernel_position_x in range(0, W, stride):
                out_y = 0
                # for each pixel in the row
                for kernel_position_y in range(0, H, stride):
                    # get a sub matrix of the input that is the same size as the filter kernel
                    sub_matrix = sample[:,
                                        kernel_position_y:kernel_position_y + HH,
                                        kernel_position_x:kernel_position_x + WW]
                    # stop if sub matrix is not full size anymore (prevent out of bounds)
                    if sub_matrix.shape != filter.shape:
                        break

                    out[sample_idx, filter_idx, out_y, out_x] = np.sum(sub_matrix * filter) + b[filter_idx]
                    out_y += 1
                out_x += 1

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################

    stride = pool_param['stride']
    pool_w = pool_param['pool_width']
    pool_h = pool_param['pool_height']

    N, C, H, W = x.shape

    h_out = 1 + (H - pool_h) // stride
    w_out = 1 + (W - pool_w) // stride

    # print('Stride: ', stride)
    # print('Input size: ', x.shape)
    # print('Pooling size W: ', pool_w)
    # print('Pooling size h: ', pool_h)
    # print('output dimensions: ', (N, C, h_out, w_out))

    # out: Output data, of shape (N, F, H', W')
    out = np.zeros((N, C, h_out, w_out))
    maxIdx = np.zeros((N, C, H, W, 2))

    # for each sample in X
    for sample_idx in range(N):
        sample = x[sample_idx]

        # iterate over sample
        for i in range(w_out):
            # calculate with stride
            pool_position_x = i * stride

            for j in range(h_out):
                pool_position_y = j * stride

                # for every channel
                for c in range(C):
                    pooling_window = sample[c,
                                            pool_position_y:pool_position_y + pool_h,
                                            pool_position_x:pool_position_x + pool_w]
                    out[sample_idx, c, j, i] = np.max(pooling_window)

                    # calculate index map
                    max_indices = np.argmax(pooling_window)
                    maxIdx[sample_idx, c, pool_position_y, pool_position_x] = np.unravel_index(max_indices,
                                                                                               pooling_window.shape)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, maxIdx, pool_param = cache

    stride = pool_param['stride']
    pool_w = pool_param['pool_width']
    pool_h = pool_param['pool_height']

    N, C, H, W = x.shape

    h_out = 1 + (H - pool_h) // stride
    w_out = 1 + (W - pool_w) // stride

    dx = np.zeros(x.shape)

    # for each sample in X
    for sample_idx in range(N):
        sample = x[sample_idx]

        # iterate over sample
        for i in range(w_out):
            # calculate with stride
            pool_position_x = i * stride

            for j in range(h_out):
                pool_position_y = j * stride

                # for every channel
                for c in range(C):
                    mc_y, mc_x = maxIdx[sample_idx, c, pool_position_y, pool_position_x]
                    dx[sample_idx,
                        c,
                        pool_position_y + int(mc_y),
                        pool_position_x + int(mc_x)] += dout[sample_idx, c, j, i]


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


import numpy as np
from dl4cv.gradient_check import (eval_numerical_gradient_array,
                                  eval_numerical_gradient,
                                  rel_error)
# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)
#
# conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_forward_naive(x, w, b, conv_param)
# correct_out = np.array([[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216 ]],
#                           [[ 0.21027089,  0.21661097],
#                            [ 0.22847626,  0.23004637]],
#                           [[ 0.50813986,  0.54309974],
#                            [ 0.64082444,  0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[ 0.69108355,  0.66880383],
#                            [ 0.59480972,  0.56776003]],
#                           [[ 2.36270298,  2.36904306],
#                            [ 2.38090835,  2.38247847]]]])
#
# # Compare your output to ours; difference should be around 1e-8
# print('Testing conv_forward_naive')
# print('difference: ', rel_error(out, correct_out))
# x_shape = (2, 3, 4, 4)
# x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
# pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
#
# out, _ = max_pool_forward_naive(x, pool_param)
#
# correct_out = np.array([[[[-0.26315789, -0.24842105],
#                           [-0.20421053, -0.18947368]],
#                          [[-0.14526316, -0.13052632],
#                           [-0.08631579, -0.07157895]],
#                          [[-0.02736842, -0.01263158],
#                           [ 0.03157895,  0.04631579]]],
#                         [[[ 0.09052632,  0.10526316],
#                           [ 0.14947368,  0.16421053]],
#                          [[ 0.20842105,  0.22315789],
#                           [ 0.26736842,  0.28210526]],
#                          [[ 0.32631579,  0.34105263],
#                           [ 0.38526316,  0.4       ]]]])
#
# # Compare your output with ours. Difference should be around 1e-8.
# print('Testing max_pool_forward_naive function:')
# print('difference: ', rel_error(out, correct_out))
#
# x = np.random.randn(3, 2, 8, 8)
# dout = np.random.randn(3, 2, 4, 4)
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
# dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
#
# out, cache = max_pool_forward_naive(x, pool_param)
# dx = max_pool_backward_naive(dout, cache)
#
# # Your error should be around 1e-12
# print('Testing max_pool_backward_naive function:')
# print('dx error: ', rel_error(dx, dx_num))