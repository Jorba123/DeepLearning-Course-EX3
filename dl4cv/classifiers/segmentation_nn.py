"""SegmentationNN"""
import torch
import torch.nn as nn


class SegmentationNN(nn.Module):

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=23, dropout=0.0):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        # input is N, C, H, W
        channels, height, width = input_dim


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
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
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
