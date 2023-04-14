import os
from matplotlib import pyplot as plt


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        os.makedirs(self.log_dir)

    def append_loss(self, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def record_accuracy(self, train_acc, val_acc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        with open(os.path.join(self.log_dir, 'train_acc.txt'), 'a') as f:
            f.write(str(train_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, 'val_acc.txt'), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")

        iters = range(len(self.train_acc))

        plt.figure()
        plt.plot(iters, self.train_acc, 'red', linewidth=2, label='train acc')
        plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='val acc')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")
