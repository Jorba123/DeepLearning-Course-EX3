"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models


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

        # get a pretrained vgg16 model
        vgg16_model = models.__dict__['vgg16'](pretrained=True)

        features = list(vgg16_model.features.children())
        classifiers = list(vgg16_model.classifier.children())

        # take the conv layers of the vgg model and put it in a nn sequential
        self.vgg_features = nn.Sequential(*features)

        # TODO: get the number of out channels of the last vgg layer
        # use the fully connected layers as convolutional layers
        fc1 = nn.Conv2d(in_channels=512,
                        out_channels=4096,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        bias=True)

        fc2 = nn.Conv2d(in_channels=4096,
                        out_channels=4096,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)

        fc3 = nn.Conv2d(in_channels=4096,
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)

        # TODO: get the number of out channels of the last vgg layer
        # Copy the data from the vgg16 net to the conv layers
        fc1.weight.data.copy_(classifiers[0].weight.data.view(4096, 512, 7, 7))
        fc1.bias.data.copy_(classifiers[0].bias.data)

        fc2.weight.data.copy_(classifiers[3].weight.data.view(4096, 4096, 1, 1))
        fc2.bias.data.copy_(classifiers[3].bias.data)

        fc3.weight.data *= weight_scale

        self.segmentation = nn.Sequential(
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            fc2,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            fc3
        )

        # perform the upsampling
        # the output has to be 23x240x240
        # output: (H_in - 1) * stride - 2 * padding + kernel size + output padding
        # 22 * 32
        self.upsampling = nn.ConvTranspose2d(in_channels=num_classes,
                                        out_channels=num_classes,
                                        kernel_size=240,
                                        stride=32,
                                        bias=False)

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
        input_size = x.size()

        x = self.vgg_features(x)
        x = self.segmentation(x)
        x = self.upsampling(x)

        # change output to match targets
        #values, indices = torch.max(x, 1)
        #indices = indices.double()[0]

        return x
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
