from queue import Queue
import sys
sys.path.append("ml_utils")

import numpy as np
import torch
from torch import manual_seed, Tensor
from torch.cuda import empty_cache
from torch.nn import Module, functional as F
from torch.optim import Optimizer, SGD

try:
    from ml_utils.data import get_data_loaders
    from ml_utils.evaluate import accuracy
    from ml_utils.model import ConvolutionalNeuralNetwork
except:
    from data import get_data_loaders
    from evaluate import accuracy
    from model import ConvolutionalNeuralNetwork

from torchvision import models
from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')

def train_step(model: Module, optimizer: Optimizer, data: Tensor,
               target: Tensor, cuda: bool):
    model.train()
    if cuda:
        data, target = data.cuda(), target.cuda()
    prediction = model(data)
    loss = F.cross_entropy(prediction, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def training(model: Module, optimizer: Optimizer, cuda: bool, n_epochs: int, 
             start_epoch: int, batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, q_stop_signal: Queue = None):
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    path = "stop.pt"
    if cuda:
        model.cuda()
    stop_signal = False
    counter = 20
    for epoch in range(start_epoch, n_epochs):
        q_epoch.put(epoch)
        for batch in train_loader:
            data, target = batch
            train_step(model=model, optimizer=optimizer, cuda=cuda, data=data,
                       target=target)
            counter += 1
            if (counter >= 20):
                counter = 0
                test_loss, test_acc = accuracy(model, test_loader, cuda)
                if q_acc is not None:
                    q_acc.put(test_acc)
                if q_loss is not None:
                    q_loss.put(test_loss)
                if q_stop_signal is not None and q_stop_signal.qsize() > 0:
                    print("queue checked")
                    stop_signal = q_stop_signal.get()
                    q_stop_signal.task_done()
        print(f"epoch{epoch} is done!")
        # print(f"epoch={epoch}, test accuracy={test_acc}, loss={test_loss}")
        if stop_signal:
            save_checkpoint(model, optimizer, epoch, test_loss, test_acc, path, False)
            #print(f"The checkpoint for epoch: {epoch} is saved!")
            print("successfully stopped")
            break
    if not stop_signal:
        save_checkpoint(model, optimizer, epoch, test_loss, test_acc, path, False)
    if cuda:
        empty_cache()

def save_checkpoint(model, optimizer, epoch, loss, acc, path, print_info):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        # Add any other information you want to save
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



def main(seed):
    print("init...")
    manual_seed(seed)
    np.random.seed(seed)
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=0.3, momentum=0.5)
    print("train...")
    training(
        model=model,
        optimizer=opt,
        cuda=False,     # change to True to run on nvidia gpu
        n_epochs=10,
        batch_size=256,
    )


if __name__ == "__main__":
    main(seed=0)
    # print(f"The final accuracy is: {final_test_acc}")

