import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import DataLoader


def accuracy(model: Module, loader: DataLoader, loss_fn: nn, cuda: bool) -> (float, float):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)

            if (str(loss_fn) == "NLLLoss()"):
                output = F.log_softmax(output, dim=1)

            if(str(loss_fn) == "MSELoss()" or str(loss_fn) == "L1Loss()"):
                target_one_hot = F.one_hot(target, 10).float()
                losses.append(loss_fn(output, target_one_hot).item())
            else:
                losses.append(loss_fn(output, target).item())

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    eval_loss = float(np.mean(losses))
    return eval_loss, 100. * correct / len(loader.dataset)


def accuracy_per_class(model: Module, loader: DataLoader, cuda: bool) \
        -> ndarray:
    model.eval()
    n_classes = len(np.unique(loader.dataset.targets))
    correct = np.zeros(n_classes, dtype=np.int64)
    wrong = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
            target = target.data.cpu().numpy().astype(np.int64)
            for label, pred in zip(target, preds):
                if label == pred:
                    correct[label] += 1
                else:
                    wrong[label] += 1
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    return 100. * correct / (correct + wrong)
