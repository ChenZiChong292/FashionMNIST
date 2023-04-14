import numpy as np
import torch
from model import Classification2DModel, Classification1DModel
from utils import confusion_matrix


def calculate_test_accuracy(dim, gen_test, num_classes):
    device = torch.device("cpu")
    Acc = []
    conf_matrix = torch.zeros(num_classes, num_classes)
    if dim == '1D':
        weight = torch.load('weight/1Dmodel.pth', map_location=device)
        model = Classification1DModel().to(device)
    elif dim == '2D':
        weight = torch.load('weight/2Dmodel.pth', map_location=device)
        model = Classification2DModel().to(device)
    else:
        print('Model dimension is wrong, ues 1D or 2D instead')
        return 0
    model.load_state_dict(weight)
    print('Start Test')
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for iteration, batch in enumerate(gen_test):
            images = batch[0]
            labels = batch[1]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
            test_acc = 100 * correct / total
            Acc.append(test_acc)
            conf_matrix = confusion_matrix(predicted, labels, conf_matrix).cpu()
        print('Finish Test')
        accuracy = np.round(np.average(np.array(Acc)),2)
    return accuracy, conf_matrix
