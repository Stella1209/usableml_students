import os
import queue
import webbrowser
import base64
import time
from PIL import Image 
import io
from itertools import chain

from io import BytesIO
from matplotlib.figure import Figure

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask import render_template, request, jsonify
import numpy as np
import torch.nn as nn
from torch import manual_seed, Tensor
from torch.optim import Optimizer, SGD
import torch
import matplotlib.pyplot as plt
#from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use('Agg')

from ml_utils.model import Adjustable_model
from ml_utils.network_drawer import Neuron, Layer, NeuralNetwork, DrawNN
from ml_utils.layer_representor import layer_box_representation
#from ml_utils.model import ConvolutionalNeuralNetwork

import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import gradio as gr
import numpy as np
import datetime
import plotly.express as px
import pandas as pd

from queue import Queue
import sys
sys.path.append("ml_utils")

import cv2
import torch

# try:
from ml_utils.data import get_data_loaders
from ml_utils.evaluate import accuracy
from ml_utils.training import load_checkpoint, prepare_training
# except:
#     from data import get_data_loaders
#     from evaluate import accuracy
#     from model import ConvolutionalNeuralNetwork

from torchvision import models
import torchvision.datasets as datasets
from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')
#import cv2

# app = Flask(__name__)
# socketio = SocketIO(app)

from torch.nn import Module, functional as F
from torch.cuda import empty_cache

# Initialize variables
seed = 42
acc = -1
loss = 0.1
text = ""
loss_fn=nn.CrossEntropyLoss()
n_epochs = 10
epoch = -1
epoch_losses = dict.fromkeys(range(n_epochs))
stop_signal = False
break_signal = False
data_image = base64.b64encode(b"").decode("ascii")
loss_img_url = f"data:image/png;base64,{data_image}"
loss_img_url = f"data:image/png;base64,{data_image}"
lr = 0.3
batch_size = 256
q_acc = queue.Queue()
q_loss = queue.Queue()
q_stop_signal = queue.Queue()
q_epoch = queue.Queue()
q_break_signal = queue.Queue()
q_text = queue.Queue()
visible_plots = gr.LinePlot()

accs = []
losses = []
epochs = []

current_model = None

# For advanced model creator:
boxes_of_layers = layer_box_representation()

"""
def prepare_training(file_name: str, 
                     n_epochs: int, 
                     start_epoch: int,
                     batch_size: int, 
                     q_acc: Queue = None, 
                     q_loss: Queue = None, 
                     q_epoch: Queue = None, 
                     q_break_signal:Queue = None,
                     q_stop_signal: Queue = None,
                     learning_rate: float = 0.001, 
                     seed: int = 42,
                     loss_fn: str = "CrossEntropyLoss"):
    manual_seed(seed)
    np.random.seed(seed)

    path = f"{file_name}.pt"
    model = Adjustable_model()
    checkpoint = load_checkpoint(model, path)
    model = Adjustable_model(linear_layers = checkpoint['lin_layers'], convolutional_layers = checkpoint['conv_layers'])
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #model = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #opt.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_fns = {"CrossEntropyLoss": nn.CrossEntropyLoss(), "NLLLoss": nn.NLLLoss(), "MSELoss": nn.MSELoss(), "L1Loss": nn.L1Loss()}
    loss_fn = loss_fns[loss_fn]

    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=start_epoch, 
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

# For advanced model creator:
boxes_of_layers = layer_box_representation()

def training(model: Module, optimizer: Optimizer, loss_fn: nn, cuda: bool, n_epochs: int, 
             start_epoch: int, batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, 
             q_break_signal:Queue = None,
             q_stop_signal: Queue = None, 
             file_name: str = None,
             lin_layers: int = 0, conv_layers: int = 0):
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    global q_text
    stop = False
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
        #q_epoch.put(epoch)
        print(f"Epoch {epoch} starts...")
        q_text.put(f"Epoch {epoch} starts...")
        path=f"{model_name}_{timestr}_{epoch}.pt"
        batch_counter = 0
        for batch in train_loader:
            data, target = batch
            
            if(str(loss_fn) == "MSELoss()" or str(loss_fn) == "L1Loss()"):
                target = F.one_hot(target, 10).float()
        
            train_step(model=model, optimizer=optimizer, loss_fn=loss_fn, cuda=cuda, data=data,
                       target=target)
            batch_counter = batch_counter+1
            if batch_counter %10 == 0:                
                print(f"{batch_counter} batches done!")
            if q_stop_signal.empty():
                continue
            if q_stop_signal.get():
                    q_break_signal.put(True)
                    stop=True
                    break
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
        q_text.put(f"epoch{epoch} is done!")
        print(f"epoch={epoch}, test accuracy={test_acc}, loss={test_loss}")
        save_checkpoint(model, optimizer, epoch, test_loss, loss_fn, test_acc, lin_layers, conv_layers, path, False)
        print(f"The checkpoint for epoch: {epoch} is saved!")
        q_text.put(f"The checkpoint for epoch: {epoch} is saved!")
        
        #print(plots)
        file = cv2.FileStorage(f"{model_name}_{timestr}_{epoch}.yml", cv2.FILE_STORAGE_WRITE)
        file.write("Plot", np.array(plots))
        file.write("Name", model_name)
        file.write("Parent", file_name)
        file.release()

        if stop:
            print("successfully stopped")
            q_text.put("successfully stopped!")
            break
    
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

def listener():
    global q_acc, q_loss, q_stop_signal, q_break_signal, q_epoch, \
    epoch, acc, loss, stop_signal, loss_img_url, epoch_losses
    while True:
        acc = q_acc.get()
        loss = q_loss.get()
        epoch = q_epoch.get()
        while((epoch_losses.get(epoch) is None) & (epoch != -1)):
            epoch_losses[epoch] = loss
        # loss_img_url = loss_plot_url()
        q_acc.task_done()
        q_loss.task_done()
        q_epoch.task_done()
"""
    
#def simple_model_creator(conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
def simple_model_creator(model_name, conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
    #global current_model
    if model_name == "":
        print("model needs a name")
        model_name = "unnamed"
    conv_layers_proto =  [{'size' : conv_layer_size, 'kernel_size' : 8, 'stride' : 2, 'padding' : 2},
                            {'size' : conv_layer_size, 'kernel_size' : 4, 'stride' : 1, 'padding' : 0},
                            {'size' : conv_layer_size, 'kernel_size' : 3, 'stride' : 1, 'padding' : 1}]
    if conv_layer_num > len(conv_layers_proto):
        conv_layers_proto = conv_layers_proto + [{'size' : conv_layer_size} for i in range(conv_layer_num - len(conv_layers_proto))]
    lin_layers = [{"linear_cells":lin_layer_size} for i in range(lin_layer_num)]
    conv_layers = [conv_layers_proto[i % 3] for i in range(conv_layer_num)]
    
    current_model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    checkpoint = {
        'epoch': 0,
        'model_state_dict': current_model.state_dict(),
        'optimizer_state_dict': current_model.state_dict(),
        'loss': 1,
        'acc': 0,
        'model_name': model_name,
        'lin_layers': lin_layers,
        'conv_layers': conv_layers
        # Add any other information you want to save
    }
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    path=f"{model_name}.pt"
    torch.save(checkpoint, path)

    file = cv2.FileStorage(f"{model_name}.yml", cv2.FILE_STORAGE_WRITE)
    file.write("Plot", np.array([]))
    file.write("Name", model_name)
    file.release()

    return make_img(conv_layer_num = conv_layer_num, lin_layer_num = lin_layer_num, conv_layer_size = conv_layer_size, lin_layer_size = lin_layer_size)

def simple_model_drawer(conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
    inp = [1]
    for i in range(conv_layer_num):
        inp.append(conv_layer_size)
    for i in range(lin_layer_num):
        inp.append(lin_layer_size)
    inp.append(10)
    print(inp)
    network = DrawNN( inp, conv_layer_num )
    return network.draw()

def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0) 
    img = Image.open(buf) 
    return img 

def make_img(conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
    fig = simple_model_drawer(conv_layer_num = conv_layer_num, lin_layer_num = lin_layer_num, conv_layer_size = conv_layer_size, lin_layer_size = lin_layer_size)
    img = fig2img(fig) 
    # Save image with the help of save() Function. 
    img.save('network.png') 
    #return os.path.join(os.path.dirname(__file__), "network.png")
    return img

# complex_model_creator:
# create base box layout (only input and output boxes) to start:
def base_boxes():
    fig = boxes_of_layers.display_current_boxes()
    img = fig2img(fig)
    img.save('layer_boxes.png')
    
base_boxes()

def add_conv_layer(convolutional_cells=32, kernel_size=3, padding=0, stride=1, output_function="Tanh", pooling="off" ):
    conv_layer = {"size" : convolutional_cells, 
             "kernel_size" : kernel_size, 
             "padding" : padding, 
             "stride" : stride, 
             "output_function" : output_function, 
             "pooling" : pooling}
    boxes_of_layers.add_conv_layer(conv_layer)
    
    fig = boxes_of_layers.display_current_boxes()
    img = fig2img(fig)
    img.save('layer_boxes.png')
    
    return img

def delete_last_conv_layer():
    if len(boxes_of_layers.get_conv_layers()) >= 1:
        boxes_of_layers.remove_conv_layer() 
    
    fig = boxes_of_layers.display_current_boxes()
    img = fig2img(fig)
    img.save('layer_boxes.png')
    
    return img

def add_lin_layer(linear_cells=32, output_function="Tanh"):
    lin_layer = {"linear_cells" : linear_cells, "output_function" : output_function}
    boxes_of_layers.add_lin_layer(lin_layer)
    
    fig = boxes_of_layers.display_current_boxes()
    img = fig2img(fig)
    img.save('layer_boxes.png')
    
    return img

def delete_last_lin_layer():
    if len(boxes_of_layers.get_lin_layers()) >= 1:
        boxes_of_layers.remove_lin_layer()
    
    fig = boxes_of_layers.display_current_boxes()
    img = fig2img(fig)
    img.save('layer_boxes.png')
    
    return img

def draw_complex_model():
    inp = [1]
    conv_layers = boxes_of_layers.get_conv_layers()
    lin_layer_dicts = boxes_of_layers.get_lin_layers()
    
    lin_layers = [i["linear_cells"] for i in lin_layer_dicts]
    
    for i in conv_layers:
        inp.append(i["size"])
    for i in lin_layers:
        inp.append(i)
    inp.append(10)
    print(inp)
    
    network = DrawNN( inp, len(conv_layers) )
    
    img = fig2img(network.draw()) 
    # Save image with the help of save() Function. 
    img.save('network.png') 
    #return os.path.join(os.path.dirname(__file__), "network.png")
    return img

def complex_model_creator(model_name):
    global current_model
    if model_name == "":
        print("model needs a name")
        model_name = "unnamed"
    
    conv_layers = boxes_of_layers.get_conv_layers()
    lin_layer_dicts = boxes_of_layers.get_lin_layers()
    
    #lin_layers = [i["linear_cells"] for i in lin_layer_dicts]
    
    current_model = Adjustable_model(linear_layers = lin_layer_dicts, convolutional_layers = conv_layers)
    checkpoint = {
        'epoch': 0,
        'model_state_dict': current_model.state_dict(),
        'optimizer_state_dict': current_model.state_dict(),
        'loss': 1,
        'acc': 0,
        'lin_layers': lin_layer_dicts,
        'conv_layers': conv_layers
        # Add any other information you want to save
    }
    path=f"{model_name}.pt"
    print(current_model)
    torch.save(checkpoint, path)

    file = cv2.FileStorage(f"{model_name}.yml", cv2.FILE_STORAGE_WRITE)
    file.write("Plot", np.array([]))
    file.write("Name", model_name)
    file.release()

    return draw_complex_model()

"""    
def start_training(model_name, seed, lr, batch_size, n_epochs, loss_fn):
    global q_acc, q_loss, stop_signal, q_stop_signal, q_break_signal, epoch, epoch_losses, loss, current_model, accs, losses, epochs
    accs, losses, epochs = [], [], []
    
    if model_name == None:
        q_text.put("Select a model first please!")

    manual_seed(seed)
    np.random.seed(seed)
    print("Starting training with:")
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"loss function: {loss_fn}")
    # execute training
    prepare_training(file_name=model_name[:(len(model_name)-3)],
             n_epochs=n_epochs,
             start_epoch = 0,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal,
             learning_rate=lr,
             seed=seed, 
             loss_fn=loss_fn)
    # return jsonify({"success": True})
"""

def stop_training():
    global break_signal, q_stop_signal
    q_stop_signal.put(True)
    print("Stopping training and finishing the training of the last epoch...")
    q_text.put("Stopping training and finishing the training of the last epoch...")
    
"""
def resume_training(model_name, seed, lr, batch_size, n_epochs, loss_fn):
    global break_signal, epoch, q_acc, q_loss, q_epoch, q_stop_signal, accs, losses, epochs
    accs, losses, epochs = [], [], []
    manual_seed(seed)
    np.random.seed(seed)
    break_signal = False
    print(f"Resume from epoch {epoch+1}")
    q_text.put(f"Resume from epoch {epoch+1}")
    if epoch != -1:
        print(f"Epoch {epoch} loaded, ready to resume training!")
        q_text.put(f"Epoch {epoch} loaded, ready to resume training!")
        prepare_training(file_name=model_name[:(len(model_name)-3)],
             n_epochs=n_epochs,
             start_epoch=epoch+1,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal,
             learning_rate=lr,
             seed=seed, 
             loss_fn=loss_fn)
    else:
        prepare_training(file_name=model_name[:(len(model_name)-3)],
             n_epochs=n_epochs,
             start_epoch=0,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal,
             learning_rate=lr,
             seed=seed, 
             loss_fn=loss_fn)
    return gr.update()
"""

def get_lates_model_file():
    app_dir = os.path.dirname(os.path.abspath(__file__)) 
    pt_files = []
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    if len(pt_files) > 0:
        last_model = max(pt_files, key=os.path.getctime)
        print(last_model)
        return last_model
    else:
        return None


def new_resume_training(model_name, seed, lr, batch_size, n_epochs, loss_fn):
    if model_name == None:
        model_name = get_lates_model_file()
        select_model.value = model_name
        if model_name == None:
            q_text.put("Found no model files. Go to the section 'Train/Test' > 'Create Model' and create a model to proceed.")
        else:
            q_text.put(f"No model selected. Starting/Resuming Training on the last created/trained model '{os.path.basename(model_name)}'.")
    q_text.put(f"Training the model '{os.path.basename(model_name)}'.")

    global q_acc, q_loss, stop_signal, q_stop_signal, q_break_signal, epoch, epoch_losses, loss, current_model, accs, losses, epochs
    accs, losses, epochs = [], [], []
    manual_seed(seed)
    np.random.seed(seed)
    print("Starting training with:")
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    prepare_training(file_name=model_name[:(len(model_name)-3)],
             n_epochs=n_epochs,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal,
             learning_rate=lr,
             seed=seed, 
             loss_fn=loss_fn)
    print("Finished Training.")
    q_text.put("Finished Training.")
    return gr.update() #jsonify({"success": True})

def revert_to_last_epoch():
    global break_signal, epoch, loss, lr, q_epoch
    print("We're at revert")
    q_text.put("Reverting to last epoch...")
    # check if the training is already stopped, if not, stop first
    if not break_signal:
        q_stop_signal.put(True)
        break_signal = q_break_signal.get(block=True)
        if break_signal:
            print("Stopping training and finishing the training of the last epoch.")
            q_text.put("Stopping training and finishing the training of the last epoch.")
    time.sleep(10)
    try:
        q_epoch.put(epoch-1) 
        q_loss.put(epoch_losses[epoch-1]) 
        loss = q_loss.get()
        epoch = q_epoch.get() 
        for i in range(epoch+1, n_epochs):
            while epoch_losses.get(i) is not None:
                epoch_losses[i] = None
        print(f"After revert epoch is {epoch}")
        q_text.put(f"After revert epoch is {epoch}")
        print(f"current epoch_losses:{epoch_losses}")
    # call loss_plot to draw the new plot
    except:
        print("You couldn't revert from epoch 0! You can resume(restart) from epoch 0 now.")
        q_text.put("You couldn't revert from epoch 0! You can resume(restart) from epoch 0 now.")
    #loss_img_url = loss_plot_url()
    return gr.update()

def get_statistics():
    global loss, q_loss, acc, q_acc, epoch, q_epoch, accs, losses, epochs
    if q_loss is not None and q_loss.qsize() > 0:
        loss = q_loss.get()
        q_loss.task_done()
        losses.append(loss)
    if q_acc is not None and q_acc.qsize() > 0:
        acc = q_acc.get()
        q_acc.task_done()
        accs.append(acc)
    if q_epoch is not None and q_epoch.qsize() > 0:
        epoch = q_epoch.get()
        q_epoch.task_done()
        epochs.append(epoch)
    return f"""
    Epoch: &emsp; &emsp; {int(epoch)}<br />
    Accuracy: &emsp; {acc}<br />
    Loss: &emsp; &emsp; &emsp; {loss}
"""
#str("Epoch:         " + str(epoch) + "\n" + "Accuracy:      " + str(acc) + "\n" + "Loss:          " + str(loss))

def get_text():
    global q_text, text
    if q_text is not None and q_text.qsize() > 0:
        text = q_text.get()
        q_text.task_done()
    return f"""{text}"""

labels_rp, epochs_rp, values_rp = [], [], []
labels_ap, epochs_ap, values_ap = [], [], []

def make_plot():
    global accs, losses, epochs, labels_rp, epochs_rp, values_rp, labels_ap, epochs_ap, values_ap
    max_len = min([len(accs), len(losses), len(epochs)])
    plot = gr.LinePlot(value=pd.DataFrame({"Labels": np.concatenate([np.array(["Accuracy - Current Run" for _ in range(max_len)] + ["Loss - Current Run" for _ in range(max_len)]), labels_rp]), 
                                           "Values": np.concatenate([np.array(accs[:max_len] + losses[:max_len]), values_rp]), 
                                           "Epochs": np.concatenate([np.array(epochs[:max_len] + epochs[:max_len]), epochs_rp])}), 
                                           x="Epochs", y="Values", color="Labels")
    return plot

def load_graph(file_names):
    global labels_rp, epochs_rp, values_rp
    labels_rp, epochs_rp, values_rp = [], [], []
    for file_name in file_names:
        file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        plots = np.array(file.getNode("Plot").mat())
        if plots.size == 1:
            plots = np.empty((3, 0), float)
        data_points = len(plots[0])
        basename = os.path.basename(file_name)
        labels_rp = np.append(labels_rp, np.concatenate([[f"Loss - {basename}" for _ in range(data_points)], [f"Accuracy - {basename}" for _ in range(data_points)]]))
        epochs_rp = np.append(epochs_rp, np.concatenate([plots[0], plots[0]]))
        values_rp = np.append(values_rp, np.concatenate([plots[1], plots[2]]))


def predict(path, img):
    if path == None:
        path = get_lates_model_file()
        select_model.value = path
        if path == None:
            q_text.put("Found no model files. Go to the section 'Train/Test' > 'Create Model' and create a model to proceed.")
        else:
            q_text.put(f"No model selected. Run prediction on the last created/trained model '{os.path.basename(path)}'.")

    img = img['composite']

    new_img = []
    for x in range(len(img)):
        new_img.append([])
        for y in range(len(img[x])):
            new_img[x].append(img[x][y][3])
    img = new_img
    
    model = Adjustable_model()
    checkpoint = load_checkpoint(model, path)
    model = Adjustable_model(linear_layers = checkpoint['lin_layers'], convolutional_layers = checkpoint['conv_layers'])
    model.load_state_dict(checkpoint['model_state_dict'])

    img = np.array(cv2.resize(np.array(img).astype('uint8'), (28,28)))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    prediction = model(img_tensor).data
    pred_out, pred_index = torch.max(prediction, 1)
    return pred_index.item()

def aaa():
    return np.zeros((28,28))

def bbb():
    return gr.update(visible=True)

visibleee = True
embed_html = '<iframe width="560" height="315" src="https://www.youtube.com/embed/bfmFfD2RIcg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'

def clear_saved_files():
    app_dir = os.path.dirname(os.path.abspath(__file__)) 
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith(".pt") or (file.endswith(".yml") and file != "env.yml"):
                os.remove(os.path.join(root, file))

with gr.Blocks() as demo:
    
    with gr.Tab("Train/Test"):
        with gr.Row():
            with gr.Column():
                with gr.Tab("Select Model"):
                    gr.Markdown("<h1>Select Model</h1>")
                    gr.Markdown("Select an already created or trained model to train or test it.")

                    button_refresh = gr.Button(value="Refresh File Explorers")
                    button_clear = gr.Button(value="Clear all saved files")
                    button_clear.click(clear_saved_files)

                    gr.Markdown("Your models will be stored as files every epoch. Because you can revert to an earlier epoch and therefore e.g. train the second epoch multiple times, the filename format is <b>modelName_date-time_lastEpoch.pt</b> (date and time refer to the point in time button start was clicked). <br/> Untrained models do not have an epoch number at the end.")

                    select_model = gr.FileExplorer("**/*.pt", label="Select Model", file_count="single", interactive=True, show_label=False)
                    button_refresh.click(None, js="window.location.reload()")
                
                with gr.Tab("Create Model"):                    
                    with gr.Tab("Beginner Model Creator"):
                        gr.Markdown("Here you can create a new model. Once a model has been created, its structure can no longer be changed. A new model must be created for that purpose.")
                        in_model_name = gr.Textbox(label="Model Name", value="unnamed")
                        in_convolutional_layers = gr.Slider(label="Convolutional Layers", value=2, minimum=0, maximum=5, step=1, info="extract features and patterns from input data. Many layers can lead to better accuracy but lengthen the training duration. ") 
                        in_cells_per_conv = gr.Slider(label="Cells per convolutional layer", value=32, minimum=1, maximum=128, step=1, info="influence the capacity and learning ability of the neural network")               
                        in_linear_layers = gr.Slider(label="Linear Layers", value=1, minimum=0, maximum=5, step=1, info="commonly used for learning complex relationships between features extracted by convolutional layers")
                        in_cells_per_lin = gr.Slider(label="Cells per linear layer", value=32, minimum=1, maximum=128, step=1, info="influence the capacity and learning ability of the neural network")
                        button_display = gr.Button(value="Display Model")
                        button_create_model = gr.Button(value="Create Model")                        
                        network_img = gr.Image(type='filepath', value='network.png', show_label=False)
                        button_create_model.click(simple_model_creator, inputs=[in_model_name, in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=network_img)
                        button_display.click(make_img, inputs = [in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=network_img)     
                        
                    with gr.Tab("Advanced Model Creator"):
                        gr.Markdown("Only recommended to people with a good understanding of CNNs.")
                        in_model_name = gr.Textbox(label="Model Name", value="unnamed")
                        with gr.Column():   
                            gr.Markdown("Add Convolutional Layer")                         
                            in_conv_cells = gr.Slider(label="Cells of convolutional layer", value=32, minimum=1, maximum=128, step=1)
                            in_kernel_size = gr.Slider(label="Kernel size", value=3, minimum=2, maximum=9, step=1)
                            in_padding = gr.Slider(label="Padding", value=0, minimum=0, maximum=5, step=1)
                            in_stride = gr.Slider(label="Stride", value=1, minimum=1, maximum=7, step=1)
                            in_conv_output_fct = gr.Dropdown(["Tanh", "Softmax", "ReLu"], label="Output Function", value="Tanh",
                                                        info="Sticking to one output function recommended")
                            in_2Dpooling = gr.Dropdown(["Off", "2", "3", "4", "5"], label="2D Pooling", value="Off")
                        with gr.Row():
                            gr.Markdown("Add Linear Layer")
                            in_lin_cells = gr.Slider(label="Cells of linear layer", value=32, minimum=1, maximum=128, step=1)
                            in_lin_output_fct = gr.Dropdown(["Tanh", "Softmax", "ReLu"], label="Output Function", value="Tanh",
                                                        info="Sticking to one output function recommended")
                        with gr.Row():    
                            button_add_conv_layer = gr.Button(value="Add Convolutional Layer")
                            button_delete_conv_layer = gr.Button(value="Remove Last Convolutional Layer")
                            button_add_lin_layer = gr.Button(value="Add Linear Layer")
                            button_delete_lin_layer = gr.Button(value="Remove Last Linear Layer")
                            layer_box_img = gr.Image(type='filepath', value='layer_boxes.png')
                            button_add_conv_layer.click(add_conv_layer, inputs=[in_conv_cells, in_kernel_size, in_padding, in_stride, in_conv_output_fct, in_2Dpooling], outputs=layer_box_img)
                            button_delete_conv_layer.click(delete_last_conv_layer, outputs=layer_box_img)
                            button_add_lin_layer.click(add_lin_layer, inputs=[in_lin_cells, in_lin_output_fct], outputs=layer_box_img)
                            button_delete_lin_layer.click(delete_last_lin_layer, outputs=layer_box_img)
                        button_complex_create_model = gr.Button(value="Create Model")
                        network_img = gr.Image(type='filepath', value='network.png', show_label=False)
                        button_complex_create_model.click(complex_model_creator, inputs=[in_model_name], outputs=network_img)


            with gr.Column():
                with gr.Tab("Adjustable Parameters"):
                    gr.Markdown("<h1>Adjustable Parameters</h1>")
                    with gr.Row():
                        with gr.Column(min_width=50):
                            pass
                        with gr.Column(min_width=50):
                            spinner = gr.Image(type='filepath', value='lama.png', label="Training...", scale=1, show_download_button=False, container=False, show_label=False)
                        with gr.Column(min_width=50):
                            pass
                    with gr.Row():
                        with gr.Tab("slider"):
                            in_learning_rate = gr.Slider(label="Learning Rate", value=0.3, minimum=0, maximum=1, step=0.01)
                        with gr.Tab("info"):
                            gr.Markdown("The <b>learning rate</b> determines how large the steps are that a model takes during training. A higher learning rate can make the model converge faster but risks overshooting the optimal solution. A lower learning rate may lead to slower convergence but can offer better accuracy. <br> As a rule, higher values are used at the beginning, the training process is interrupted in between and smaller values are then selected.")
                    with gr.Row():
                        with gr.Tab("slider"):
                            in_batch_size = gr.Slider(label="Batch Size", value=256, minimum=0, maximum=1024, step=32)
                        with gr.Tab("info"):
                            gr.Markdown("<b>Batch size</b> refers to the number of training examples processed in one epoch. It affects the stability of the training process, memory usage, and generalization ability. A high value leads to better accuracy, but prolongs the training process.")
                    with gr.Row():
                        with gr.Tab("slider"):
                            in_seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=1000, step=1)
                        with gr.Tab("info"):
                            gr.Markdown("The <b>seed</b> has no direct influence on accuracy or training duration. It roughly allows you to achieve different results when training with the same parameters.")
                    with gr.Row():
                        with gr.Tab("slider"):
                            in_n_epochs = gr.Slider(label="Epochs/Training Steps", value=10, minimum=0, maximum=50, step=1)
                        with gr.Tab("info"):
                            gr.Markdown("<b>Epochs</b> refer to the number of times a learning algorithm sees the entire dataset during the training process. It's a hyperparameter that controls the number of iterations the algorithm makes over the entire dataset. Though more epochs extend the training process, they allow the model to learn more from the dataset and potentially improve its performance.")
                    with gr.Row():
                        with gr.Tab("slider"):   
                            in_loss_fn = gr.Dropdown(label="Loss Function", value="CrossEntropyLoss", choices=["CrossEntropyLoss", "NLLLoss", "MSELoss", "L1Loss"])
                        with gr.Tab("info"):
                            gr.Markdown("A <b>loss function</b> measures how well a model performs on a dataset by comparing its predictions to the actual target values. It quantifies the difference between predicted outputs and ground truth labels. The goal during training is to minimize this difference.")
                    
            with gr.Column():
                with gr.Tab("Training"):
                    gr.Markdown("<h1>Training</h1>")
                    with gr.Row():
                        with gr.Column(min_width=100):
                            button_start = gr.Button(value="Start/Resume")
                        with gr.Column(min_width=100):
                            button_stop = gr.Button(value="Stop")
#                        with gr.Column(min_width=100):
#                           button_resume = gr.Button(value="Continue")
                    with gr.Row():
                        button_revert = gr.Button(value="Revert to last epoch")
                        button_revert.click(revert_to_last_epoch, inputs=None, outputs=None)
                    with gr.Row():
                        text_component = gr.Markdown()
            
                    #button_start.click(bbb, inputs=None, outputs=spinner)
                    button_start.click(new_resume_training, inputs=[select_model, in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_loss_fn], outputs=spinner)
                    button_stop.click(stop_training, inputs=None, outputs=None)
                    #button_resume.click(resume_training,inputs=[select_model, in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_loss_fn], outputs=spinner)

                    training_plot = gr.LinePlot(show_label=False)
                    training_info = gr.Markdown()
                    gr.Markdown("Choose which models you want to display in the plot:")
                    select_plot = gr.FileExplorer("**/*.yml", file_count="multiple", show_label=False)
                    select_plot.change(load_graph, inputs=[select_plot], outputs=[])
                with gr.Tab("Testing"):
                    gr.Markdown("<h1>Testing</h1>")
                    gr.Markdown("Here you can test if the trained model recognizes the number you draw.")
                    buttoton = gr.Button(value="Activate & start Sketechpad")
                    playground_in = gr.Sketchpad(value=np.zeros((28,28)), crop_size=("1:1"), type="numpy", interactive=True)
                    button_test = gr.Button(value="Test")
                    playground_out = gr.Text(label="Result")
                    button_test.click(predict, inputs=[select_model, playground_in], outputs=[playground_out])
                    buttoton.click(aaa, inputs=None, outputs=[playground_in])

    with gr.Tab("Info"):
        gr.Markdown("<h1 align='center'>Introduction to Machine Learning</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=7):
                gr.HTML(embed_html)           
                gr.Markdown(
                """
                <span style="font-weight:300;font-size:20px;text-align:justify">

                In order to explain the term machine learning, we must first deal with the term artificial intelligence. 
                Artificial intelligence is a scientific discipline that focuses on the research and algorithmization of 
                preferably human intelligence in the form of automatically usable perception and "mind power". 
                Artificial intelligence (AI for short) is therefore a machine that can replicate the cognitive abilities of a human being, 
                i.e. automates human intelligence. Philosophers and psychologists have been discussing what exactly 
                intelligence is for thousands of years, but the ability to learn is a generally recognized component.\n

                This brings us to the next term, "machine learning". Just as a person only becomes intelligent through lifelong learning, 
                a machine only becomes intelligent through learning processes. The advantage of using such processes is that machines learn 
                independently how to solve certain problems. This becomes particularly advantageous if the problem cannot be described in concrete 
                terms or demonstrates such variability that a clearly definable solution proves challenging to pinpoint.\n\n

                Trainable programs are often implemented in the form of neural networks. Neural networks are a set of algorithms, 
                modeled loosely after the human brain, that are designed to recognize patterns.
                Neurons within neural networks can be envisioned as interconnected information nodes. They receive input data and process it, 
                aiming to generate an output that closely matches the desired result or ideally achieves it precisely.
                To simplify, think of neurons as tiny decision-makers. When the network achieves a good outcome, these neurons adjust 
                themselves to provide more similar decisions in the future. However, if the outcome is not satisfactory, 
                the neurons recalibrate to offer different decisions /output next time. 
                It's like refining the network's judgment based on whether it's geting things right or wrong.</span>
                """)

                gr.Image("NN.png",width=450, show_label=False) #"https://www.researchgate.net/profile/Anna-Meiliana/publication/334845867/figure/fig1/AS:787149469270017@1564682466919/A-deep-neural-network-simplified40-Adapted-with-permission-from-Springer-Nature.png"

                gr.Markdown(
                """
                <span style="font-weight:300;font-size:20px;text-align:justify">
                This tool focuses on image recognition with neural networks. The data, the tool works with, is the MNIST-dataset, 
                which is a collection of handwritten digits from 0 to 9 widely used for training various machine learning models. 
                Each image is labeled with the number it shows. The trained models should be able to classifiy the images and recognize the displayed number.
                Neural networks, that are used specifically to learn grid-like data such as images, are called Convolutional Neural Networks (CNN). 
                Image pixels contain values that indicate the color each pixel should be. 
                The MNIST images are grayscale, therefore each pixel has a value between 0 (black) and 255 (white).
                </span>"""
                )

                gr.Image("pixel-values-matrix.png",width=450, show_label=False) #"source:https://www.researchgate.net/figure/Representation-of-value-three-in-the-MNIST-dataset-and-its-equivalent-matrix_fig1_361444345"

                gr.Markdown(
                """
                <span style="font-weight:300;font-size:20px;text-align:justify">
                These pixel values serve as input to the CNN. Simplified, they are processed by several convolutional layers followed by linear layers.
                The convolutional layers detect features/patterns in data. The layers are structured to initially recognize simpler patterns like lines and curves, 
                progressing to identify more complex patterns such as faces and objects as they advance. The deeper (more convLayers) the better the pattern 
                recognition, but it also extends the training duration and processing load.
                </span>"""
                )
                gr.Image("hierarchy.png",width=450, show_label=False) #https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/41/0f/hierarchy.png

                gr.Markdown(
                """
                <span style="font-weight:300;font-size:20px;text-align:justify">
                The linear layers aid in determining the correct class for the image, which corresponds to the digits 0 through 9.
                In the end the predicted output result of the model is compared to the actual target labels. 
                A loss function (such as cross-entropy loss) is used to quantify the difference between the predicted output and the true labels. 
                Based on the loss it changes learnable parameters in the neurons accordingly and repeats the process.
                </span>
                """
                )

                #gr.Image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*XdCMCaHPt-pqtEibUfAnNw.png",width=450, show_label=False) #https://miro.medium.com/v2/resize:fit:1400/format:webp/1*XdCMCaHPt-pqtEibUfAnNw.png
                gr.Markdown("""More detailed resources: 
                            https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
                            https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac
                            """)
                
            with gr.Column(scale=1):
                pass
        """ with gr.Row():
            with gr.Column(min_width=50):
                pass
            with gr.Column(min_width=50):
                gr.HTML(embed_html)
                #gr.Video("https://www.youtube.com/embed/bfmFfD2RIcg")
            with gr.Column(min_width=50):
                pass"""

    dep1 = demo.load(get_statistics, None, training_info, every=0.5)
    dep2 = demo.load(make_plot, None, training_plot, every=0.5)
    dep3 = demo.load(get_text, None, text_component, every=0.5)

if __name__ == "__main__":
    webbrowser.open_new_tab(f"http://127.0.0.1:7860/")
    demo.queue().launch()