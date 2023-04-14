from matplotlib import pyplot as plt


def confusion_matrix(prediction, labels, conf_matrix):
    for p, t in zip(prediction, labels):
        conf_matrix[p, t] = conf_matrix[p, t] + 1
    return conf_matrix


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[labels]
