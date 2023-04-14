import os

import numpy as np
import torch
from tqdm import tqdm


def fit_one_epoch(model_train, model, classifier_loss, optimizer, gen, gen_val, cuda, local_rank,
                  epoch, epoch_step, epoch_step_val, Epoch, loss_history, save_period, save_dir):
    loss = 0
    val_loss = 0
    print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    correct = 0
    total = 0
    train_acc = 0
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images = batch[0]
        labels = batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = [ann.cuda(local_rank) for ann in labels]
        optimizer.zero_grad()
        outputs = model_train(images)
        loss_value_all = 0
        loss_item = classifier_loss(outputs, labels)
        loss_value_all = loss_value_all + loss_item
        loss_value = loss_value_all
        loss_value.backward()
        optimizer.step()
        loss = loss + loss_value.item()

        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                            'train_acc': np.round(train_acc, 2)})
        pbar.update(1)

    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    correct = 0
    total = 0
    val_acc = 0
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images = batch[0]
        labels = batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = [ann.cuda(local_rank) for ann in labels]
        optimizer.zero_grad()
        outputs = model_train(images)
        loss_value_all = 0
        loss_item = classifier_loss(outputs, labels)
        loss_value_all += loss_item
        loss_value = loss_value_all
        val_loss = val_loss + loss_value.item()
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                            'val_acc': np.round(val_acc, 2)})
        pbar.update(1)

    pbar.close()
    print('Finish Validation')
    loss_history.record_accuracy(train_acc, val_acc)
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))