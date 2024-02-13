from queue import Queue
import sys
import time
sys.path.append("ml_utils")

import numpy as np
import torch
from torch import manual_seed, Tensor
from torch.cuda import empty_cache
import torch.nn as nn
from torch.nn import Module, functional as F
from torch.optim import Optimizer, SGD

try:
    from ml_utils.data import get_data_loaders
    from ml_utils.evaluate import accuracy
    from ml_utils.model import Adjustable_model
except:
    from data import get_data_loaders
    from evaluate import accuracy
    from model import Adjustable_model

from torchvision import models
from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')

import cv2

def prepare_training(file_name: str, n_epochs: int, 
             batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, 
             q_break_signal:Queue = None,
             q_stop_signal: Queue = None,
             learning_rate: float = 0.001, 
             seed: int = 42,
             loss_fn: str = "CrossEntropyLoss"):
    
    manual_seed(seed)
    np.random.seed(seed)

    path = f"{file_name}.pt"
    if q_stop_signal is not None:
        q_stop_signal.put(False)
    print(f"Resume from epoch {n_epochs}")
    #path = f"stop{n_epochs}.pt"
    model = Adjustable_model()
    #model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    checkpoint = load_checkpoint(model, path)
    model = Adjustable_model(linear_layers = checkpoint['lin_layers'], convolutional_layers = checkpoint['conv_layers'])
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #model = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Epoch {n_epochs} loaded, ready to resume training!")

    loss_fns = {"CrossEntropyLoss": nn.CrossEntropyLoss(), "NLLLoss": nn.NLLLoss(), "MSELoss": nn.MSELoss(), "L1Loss": nn.L1Loss()}
    loss_fn = loss_fns[loss_fn]

    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=checkpoint['epoch']+1,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal, 
             file_name=file_name, lin_layers = checkpoint['lin_layers'], conv_layers = checkpoint['conv_layers'],
             loss_fn=loss_fn)
    


def train_step(model: Module, optimizer: Optimizer, loss_fn: nn, data: Tensor,
               target: Tensor, cuda: bool):
    model.train()
    if cuda:
        data, target = data.cuda(), target.cuda()
    prediction = model(data)

    if (str(loss_fn) == "NLLLoss()"):
        prediction = F.log_softmax(prediction, dim=1)

    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def training(model: Module, optimizer: Optimizer, loss_fn: nn, cuda: bool, n_epochs: int, 
             start_epoch: int, batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, 
             q_break_signal:Queue = None,
             stop_signal: bool = False, 
             q_stop_signal: Queue = None, 
             file_name: str = None,
             lin_layers: int = 0, conv_layers: int = 0):
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    if cuda:
        model.cuda()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    file = cv2.FileStorage(f"{file_name}.yml", cv2.FILE_STORAGE_READ)
    model_name = file.getNode("Name").string()
    plots = np.array(file.getNode("Plot").mat())
    if plots.size == 1:
        plots = np.empty((3, 0), float)
    if q_acc is not None:
        for acc in plots[2]:
            q_acc.put(acc)
    if q_loss is not None:
        for loss in plots[1]:
            q_loss.put(loss)
    if q_epoch is not None:
        for epoch in plots[0]:
            q_epoch.put(epoch)

    for epoch in range(start_epoch, n_epochs):

        if not q_stop_signal.empty():
            if q_stop_signal.get():
                with q_stop_signal.mutex:
                    q_stop_signal.queue.clear()
                q_break_signal.put(True)
                break

        print(f"Epoch {epoch} starts...")
        path=f"{model_name}_{timestr}_{epoch}.pt"
        for batch in train_loader:
            data, target = batch

            if(str(loss_fn) == "MSELoss()" or str(loss_fn) == "L1Loss()"):
                target = F.one_hot(target, 10).float()
        
            train_step(model=model, optimizer=optimizer, loss_fn=loss_fn, cuda=cuda, data=data,
                       target=target)
        test_loss, test_acc = accuracy(model, test_loader, loss_fn, cuda)
        if q_acc is not None:
            q_acc.put(test_acc)
        if q_loss is not None:
            q_loss.put(test_loss)
        if q_epoch is not None:
            q_epoch.put(epoch)
        #print(plots)
        plots = np.append(plots, np.array([[epoch, test_loss, test_acc]]).transpose(), axis=1)
        print(f"epoch{epoch} is done!")
        print(f"epoch={epoch}, test accuracy={test_acc}, loss={test_loss}")
        save_checkpoint(model, optimizer, epoch, test_loss, loss_fn, test_acc, lin_layers, conv_layers, path, False)
        print(f"The checkpoint for epoch: {epoch} is saved!")

        file = cv2.FileStorage(f"{model_name}_{timestr}_{epoch}.yml", cv2.FILE_STORAGE_WRITE)
        file.write("Plot", np.array(plots))
        file.write("Name", model_name)
        file.write("Parent", file_name)
        file.release()

    if cuda:
        empty_cache()

def save_checkpoint(model, optimizer, epoch, loss, acc, loss_fn, lin_layers, conv_layers, path, print_info):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'loss_fn': loss_fn,
        # Add any other information you want to save
        'lin_layers': lin_layers,
        'conv_layers': conv_layers
    }
    torch.save(checkpoint, path)
    if(print_info):
        print(f"The checkpoint for epoch: {epoch} is saved!")
            # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print()
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        
def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.eval()
    return checkpoint
