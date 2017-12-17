import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from dl4cv.classifiers.classification_cnn import ClassificationCNN
from dl4cv.solver import Solver
from dl4cv.data_utils import get_CIFAR10_datasets, OverfitSampler, rel_error
if __name__ == '__main__':
    train_data, val_data, test_data, mean_image = get_CIFAR10_datasets()

    #torch.manual_seed(0)
    #np.random.seed(0)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False, num_workers=4)

    model = ClassificationCNN()
    solver = Solver(optim_args={"lr": 1e-2})
    solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=10)
    print('CUDA active: ',model.is_cuda)

    # from dl4cv.classifiers.segmentation_nn import SegmentationNN
    # import torch.nn.functional as F
    # from dl4cv.data_utils import OverfitSampler
    # from dl4cv.data_utils import SegmentationData, label_img_to_rgb
    #
    # path_to_dataset = '/Users/felix/Documents/Repositories/TUM/DL4CV/03/DeepLearning-Course-EX3/datasets/segmentation_data/'
    # path_to_dataset = 'C:\\Users\\felix\\OneDrive\\Studium\\Studium\\4. Semester\\DL4CV\\Exercises\\03\\dl4cv\\exercise_3\\datasets\\segmentation_data\\'
    # train_data = SegmentationData(image_paths_file=path_to_dataset + 'train.txt')
    # val_data = SegmentationData(image_paths_file=path_to_dataset + 'val.txt')
    # test_data = SegmentationData(image_paths_file= 'C:\\Users\\felix\\OneDrive\\Studium\\Studium\\4. Semester\\DL4CV\\Exercises\\03\\dl4cv\\exercise_3\\datasets\\segmentation_data_test\\test.txt')
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=1,
    #                                           shuffle=False,
    #                                           num_workers=1)
    #
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False, num_workers=4,
    #                                            sampler=OverfitSampler(1))
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, num_workers=4)
    #
    # overfit_model = SegmentationNN()
    # print(overfit_model)
    # overfit_solver = Solver(optim_args={"lr": 1e-2}, loss_func=torch.nn.CrossEntropyLoss(ignore_index=-1))
    # overfit_solver.train(overfit_model, train_loader, val_loader, log_nth=1, num_epochs=10)




