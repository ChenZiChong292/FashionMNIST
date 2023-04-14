import os
import datetime
import numpy as np
from torch import nn
from torchvision import transforms
from callbacks import LossHistory
from model import Classification2DModel
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from test import calculate_test_accuracy
from utils_fit import fit_one_epoch


def model2d(mode):
    image_size = [112, 112]
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize([image_size[0], image_size[1]]),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.FashionMNIST(root='', train=False, transform=transform, download=True)
    data_size = 5000  # choose 5000 images from the whole dataset
    other_data_size = 5000
    # choose 3500 images as train_set, choose 1000 images as val_set, choose 500 images as test_set,
    data_set, other_data_set = random_split(dataset, [data_size, other_data_size],
                                            generator=torch.Generator().manual_seed(0))
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = int(data_size * 0.1)

    train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(0))
    train_batch_size = 32
    test_batch_size = 8
    val_batch_size = 16
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    num_test = len(test_dataset)
    shuffle = True
    nw = 0
    drop_last = True
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                              shuffle=shuffle, num_workers=nw, drop_last=drop_last)
    test_loader = DataLoader(dataset=val_dataset, batch_size=test_batch_size,
                             shuffle=shuffle, num_workers=nw, drop_last=drop_last)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size,
                            shuffle=shuffle, num_workers=nw, drop_last=drop_last)

    model = Classification2DModel()
    model_train = model.train()
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 50
    cuda = False
    local_rank = 0
    save_period = 10
    epoch_step = num_train // train_batch_size
    epoch_step_test = num_test // test_batch_size
    epoch_step_val = num_val // val_batch_size
    if mode == 'train':
        save_dir = 'logs'
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir)
        for e in range(epoch):
            fit_one_epoch(model_train, model, loss_function, optimizer, train_loader, val_loader, cuda, local_rank,
                          e, epoch_step, epoch_step_val, epoch, loss_history, save_period, save_dir)
    elif mode == 'test':
        accuracy_2D, conf_mat = calculate_test_accuracy('2D', test_loader, 10)
        print('The accuracy of 2Dmodel on test dataset is: ' + str(accuracy_2D) + '%')
        print('The confusion matrices of 2Dmodel on test dataset is as follows:')
        print(np.array(conf_mat))
    else:
        print('mode is wrong, use train or test instead')
