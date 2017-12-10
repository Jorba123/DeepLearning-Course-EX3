"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        # Layer 1 Conf
        # Padding => Same
        conv_padding = (kernel_size - stride_conv) // 2
        conv_out = (num_filters,
                    (((width + 2 * conv_padding - kernel_size) // stride_conv) + 1),
                    (((height + 2 * conv_padding - kernel_size) // stride_conv) + 1))

        # Layer 3 Pool
        pool_out = (conv_out[0],
                    ((conv_out[1] - pool) // stride_pool + 1),
                    ((conv_out[2] - pool) // stride_pool + 1))

        # Layer 4 Fully Connected
        fc1_dim = ((pool_out[0] * pool_out[1] * pool_out[2]), hidden_dim)

        # Layer 7 Fully Connected
        fc2_dim = (hidden_dim, num_classes)

        print('Conv Pad ', conv_padding)
        print('Conv Out ', conv_out)
        print('Pool Out ', pool_out)
        print('FC1 ', fc1_dim)
        print('FC2 ', fc2_dim)

        # conv - relu - 2x2 max pool - fc - dropout - relu - fc
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride_conv,
                padding=conv_padding,
                bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=stride_pool)
        )

        # initialize and scale layers
        # get weights of first conv layer
        conv_layer = self.features.children().__next__()
        conv_layer.weight.data *= weight_scale

        self.features_out = pool_out

        self.classifier = nn.Sequential(
            nn.Linear(in_features=fc1_dim[0], out_features=fc1_dim[1]),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=fc2_dim[0], out_features=fc2_dim[1])
        )


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        x = self.features(x)
        x = x.view(x.size(0), self.features_out[0] * self.features_out[1] * self.features_out[2])
        x = self.classifier(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
