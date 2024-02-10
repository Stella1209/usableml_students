import os
#os.system("pip uninstall -y gradio")
#os.system("pip install gradio==3.50.2")
import queue
import webbrowser
import base64
import time
from PIL import Image 
import io
import os

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
from ml_utils.training import training, load_checkpoint
from ml_utils.training import training, load_checkpoint, prepare_training
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

# For advanced model creator:
boxes_of_layers = layer_box_representation()

def training(model: Module, optimizer: Optimizer, cuda: bool, n_epochs: int, 
             start_epoch: int, batch_size: int, q_acc: Queue = None, q_loss: Queue = None, 
             q_epoch: Queue = None, 
             q_break_signal:Queue = None,
             q_stop_signal: Queue = None,
             progress=gr.Textbox()):
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    global q_text
    stop = False
    if cuda:
        model.cuda()
    counter = 20
    for epoch in range(start_epoch, n_epochs):
        q_epoch.put(epoch)
        print(f"Epoch {epoch} starts...")
        q_text.put(f"Epoch {epoch} starts...")
        path=f"stop{epoch}.pt"
        print(f"Epoch {epoch} in progress...")
        q_text.put(f"Epoch {epoch} in progress...")
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
                if q_stop_signal.empty():
                    continue
                if q_stop_signal.get():
                    q_break_signal.put(True)
                    stop=True
                    break
        if stop:
            print("successfully stopped")
            q_text.put("successfully stopped!")
            break
        print(f"epoch{epoch} is done!")
        q_text.put(f"epoch{epoch} is done!")
        save_checkpoint(model, optimizer, epoch, test_loss, test_acc, path, False)
        print(f"The checkpoint for epoch: {epoch} is saved!")
        q_text.put(f"The checkpoint for epoch: {epoch} is saved!")
        
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


# @app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, epoch, epoch_losses, loss_img_url, lr, n_epochs, batch_size, loss_fn
    # render "index.html" as long as user is at "/"
    return render_template("index.html", seed=seed, acc=acc, \
                           loss=loss, epoch = epoch, loss_plot = loss_img_url, 
                           lr=lr, n_epochs=n_epochs, batch_size=batch_size)
    
#def simple_model_creator(conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
def simple_model_creator(model_name, conv_layer_num = 2, lin_layer_num = 1, conv_layer_size = 32, lin_layer_size = 32):
    #global current_model
    if model_name == "":
        print("model needs a name")
        model_name = "unnamed"
    conv_layers_proto =  [{'size' : conv_layer_size, 'kernel_size' : 8, 'stride' : 2, 'padding' : 2},
                            {'size' : conv_layer_size, 'kernel_size' : 4, 'stride' : 1, 'padding' : 0}]
    if conv_layer_num > len(conv_layers_proto):
        conv_layers_proto = conv_layers_proto + [{'size' : conv_layer_size} for i in range(conv_layer_num - len(conv_layers_proto))]
    lin_layers = [lin_layer_size for i in range(lin_layer_num)]
    conv_layers = [conv_layers_proto[i % 2] for i in range(conv_layer_num)]
    
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

    return

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
    
    lin_layers = [i["linear_cells"] for i in lin_layer_dicts]
    
    current_model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    checkpoint = {
        'epoch': 0,
        'model_state_dict': current_model.state_dict(),
        'optimizer_state_dict': current_model.state_dict(),
        'loss': 1,
        'acc': 0,
        'lin_layers': lin_layers,
        'conv_layers': conv_layers
        # Add any other information you want to save
    }
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    path=f"{model_name}.pt"
    print(current_model)
    torch.save(checkpoint, path)

    file = cv2.FileStorage(f"{model_name}.yml", cv2.FILE_STORAGE_WRITE)
    file.write("Plot", np.array([]))
    file.write("Name", model_name)
    file.release()

    return draw_complex_model()

    

# def index():
#     global seed, acc, loss, epoch, loss_img_url, lr, n_epochs, batch_size
#     # render "index.html" as long as user is at "/"
#     return render_template("index.html", seed=seed, acc=acc, \
#                            loss=loss, epoch = epoch, loss_plot = loss_img_url, 
#                            lr=lr, n_epochs=n_epochs, batch_size=batch_size)

# @app.route("/start_training", methods=["POST"])
def start_training(seed, lr, batch_size, n_epochs):
    print("starting Training with seed " + str(seed))
    # ensure that these variables are the same as those outside this method
    #global q_acc, q_loss, stop_signal, epoch, epoch_losses, loss, current_model
    global q_acc, q_loss, stop_signal, q_stop_signal, q_break_signal, epoch, epoch_losses, loss, current_model, accs, losses, epochs, loss_fn
    accs, losses, epochs = [], [], []
        
    #lin_layers = [32 for i in range(lin_layer_num)]
    #conv_layers = [conv_layers_proto[i] for i in range(conv_layer_num)]
    
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = current_model
    #opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #print(learning_rate)
    #print(n_epochs)
    #print(batch_size)
    #model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    print("Starting training with:")
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"loss function: {loss_fn}")
    # execute training
    training(model=model,
             optimizer=opt,
             loss_fn=loss_fn,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=0,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal)
    # return jsonify({"success": True})

# @app.route("/stop_training", methods=["POST"])
def stop_training():
    global stop_signal, q_stop_signal
    if q_stop_signal is not None:
        q_stop_signal.put(True)
    stop_signal = True  # Set the stop signal to True
    # saveCheckpoint()
    return #gr.update(visible=False) #jsonify({"success": True})

    #global break_signal
    #if not break_signal:
    #    q_stop_signal.put(True)
        # set block to true to wait for item if the queue is empty
    #    break_signal = q_break_signal.get(block=True)
    #if break_signal:
    #    print("Training breaks!")
    #return jsonify({"success": True})
# @app.route("/resume_training", methods=["POST"])


    #lin_layers = [32 for i in range(lin_layer_num)]
    #conv_layers = [conv_layers_proto[i] for i in range(conv_layer_num)]

#@app.route("/resume_training", methods=["POST"])
"""def resume_training(seed, lr, batch_size, n_epochs):
    global break_signal, epoch, q_acc, q_loss, q_epoch, q_stop_signal, stop_signal, current_model
    manual_seed(seed)
    np.random.seed(seed)

    path = "stop.pt"
    if q_stop_signal is not None:
        q_stop_signal.put(False)
    stop_signal = False  # Set the stop signal to False
    # q_stop_signal.put(False)
    break_signal = False
    # print(f"before get, epoch is {epoch}")
    # epoch = q_epoch.get()
    print(f"Resume from epoch {epoch}")
    path = f"stop{epoch}.pt"
    model = current_model
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    checkpoint = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Epoch {epoch} loaded, ready to resume training!")
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
             q_stop_signal=q_stop_signal)
    return #jsonify({"success": True})
    global break_signal
    if not break_signal:
        q_stop_signal.put(True)
        # set block to true to wait for item if the queue is empty
        break_signal = q_break_signal.get(block=True)
    if break_signal:
        print("Training breaks!")
        q_text.put("Training breaks!")
    # return jsonify({"success": True})

# @app.route("/resume_training", methods=["POST"])
def resume_training(seed, lr, batch_size, n_epochs):
    global break_signal, epoch, q_acc, q_loss, q_epoch, q_stop_signal
    manual_seed(seed)
    np.random.seed(seed)
    break_signal = False
    print(f"Resume from epoch {epoch}")
    q_text.put(f"Resume from epoch {epoch}")
    try:
        path = f"stop{epoch-1}.pt"
        model = ConvolutionalNeuralNetwork()
        opt = SGD(model.parameters(), lr=lr, momentum=0.5)
        checkpoint = load_checkpoint(model, path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Epoch {epoch-1} loaded, ready to resume training!")
        q_text.put(f"Epoch {epoch-1} loaded, ready to resume training!")
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
                q_stop_signal=q_stop_signal)
    except:
        training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=0,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal)"""
    # return jsonify({"success": True})


def new_resume_training(model_name, seed, lr, batch_size, n_epochs, loss_fn):
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
    return gr.update(visible=False) #jsonify({"success": True})

#app.route("/revert_to_last_epoch", methods=["GET", "POST"])
def revert_to_last_epoch():
    global break_signal, epoch, loss, lr, q_epoch, loss_img_url
    # check if the training is already stopped, if not, stop first
    if not break_signal:
        q_stop_signal.put(True)
        break_signal = q_break_signal.get(block=True)
        if break_signal:
            print("Training breaks!")
            q_text.put("Training breaks!")
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
    #return jsonify({"epoch_losses": epoch_losses})

# def remember(loss):
#     global stop_signal, epoch_losses, loss_img_url, data_url, epoch
#     # to forget losses after the epoch
#     for i in range(epoch+1, n_epochs):
#         while epoch_losses.get(i) is not None:
#             epoch_losses[i] = None
#     print(f"current epoch_losses:{epoch_losses}")
#     loss_img_url = loss_plot_url()
#     while stop_signal == True:
#         q_loss.put(loss)
#         q_epoch.put(epoch)
#     return jsonify({"success": True})
    

#@app.route("/loss_plot", methods=["GET"])
# loss_plot is for the display at endpoint /loss_plot while loss_plot_2 is for the display at index.html
def loss_plot():
    global data_url
    data_full_url = f"<img src='{data_url}'/>"
    print(epoch_losses)
    return data_full_url

def loss_plot_url():
    global epoch_losses, loss, epoch, data_url
    fig = Figure()
    ax = fig.subplots()  # Create a new figure with a single subplot
    y = list(epoch_losses.values())
    ax.plot(range(epoch+1),y[:(epoch+1)])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss per Epoch')
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data_image = base64.b64encode(buf.getbuffer()).decode("ascii")
    data_url = f"data:image/png;base64,{data_image}"
    return data_url

# # @app.route("/acc_plot", methods=["GET"])
# def acc_plot():
#     # Create a Matplotlib plot
#     x = np.linspace(0, 2 * np.pi, 100)
#     y = np.sin(x)
#     # Plot the data and save the figure
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     buf = BytesIO()
#     fig.savefig(buf, format="png")
#     # Embed the result in the html output.
#     data_image = base64.b64encode(buf.getbuffer()).decode("ascii")
#     data_url = f"<img src='data:image/png;base64,{data_image}'/>"
#     return data_url
    # loss_img_url = loss_plot_url()
    # return jsonify({"epoch_losses": epoch_losses})
    

# @app.route("/update_seed", methods=["POST"])
def update_seed():
    global seed
    seed = int(request.form["seed"])
    return jsonify({"seed": seed})

#adjust learning rate 
# @app.route("/update_learningRate", methods=["POST"])
def update_learningRate():
    global lr
    lr = float(request.form["lr"])
    return jsonify({"lr": lr})

#adjust number of epochs
# @app.route("/update_numEpochs", methods=["POST"])
def update_numEpochs():
    global n_epochs
    n_epochs = int(request.form["n_epochs"])
    return jsonify({"n_epochs": n_epochs})

#adjust batch_size
# @app.route("/update_batch_size", methods=["POST"])
def update_batch_size():
    global batch_size
    batch_size = int(request.form["batch_size"])
    return jsonify({"batch_size": batch_size})

#adjust loss_fn
#@app.route("/update_loss_fn", methods=["POST"])
def update_loss_fn():
    global loss_fn
    loss_dict = {"CrossEntropy":nn.CrossEntropyLoss(),
                 "NegativeLog":nn.NLLLoss(),
                 "L1":nn.L1Loss(),
                 "MSE":nn.MSELoss()}
    loss_fn = loss_dict[str(request.form["loss_fn"])]
    return jsonify({"success": True})

#@app.route("/get_accuracy")
def get_accuracy():
    global acc
    return jsonify({"acc": acc})

# @app.route("/get_loss")
def get_loss():
    global loss
    return jsonify({"loss": loss})

# @app.route("/get_epoch")
def get_epoch():
    global epoch
    return jsonify({"epoch": epoch})

# @app.route("/get_epoch_losses")
def get_epoch_losses():
    global epoch_losses
    return jsonify({"epoch_losses": epoch_losses})

# @app.route("/get_dict")
def get_dict():
    dictTest = dict({"one": "1", "two": "2"})
    return jsonify({"dictTest": dictTest})

# # @app.route("/get_loss_image")
# def get_loss_image():
#     global loss_img_url
#     return jsonify({"loss_img_url": loss_img_url})

"""
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5001
    print("App started")
    threading.Thread(target=listener, daemon=True).start()
    webbrowser.open_new_tab(f"http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=True)
"""

def get_loss():
    global loss, q_loss
    if q_loss is not None and q_loss.qsize() > 0:
        loss = q_loss.get()
        q_loss.task_done()
    return loss

def get_accuracy():
    global acc, q_acc
    if q_acc is not None and q_acc.qsize() > 0:
        acc = q_acc.get()
        q_acc.task_done()
    return acc

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

    #training_steps = []
    max_len = min([len(accs), len(losses), len(epochs)])
    #for j in range(2):
    #    for i in range(max_len):
    #        training_steps.append(i + 1)
    #plot = gr.LinePlot(value=pd.DataFrame({"Epoch": training_steps, "Accuracy": accs, "Loss": losses}), x="Epoch", y="Accuracy")
    #plot = gr.LinePlot(value=pd.DataFrame({"Labels": ["Accuracy" for _ in range(max_len)] + ["Loss" for _ in range(max_len)], "Values": accs[:max_len] + losses[:max_len], "Epochs": training_steps}), x="Epochs", y="Values", color="Labels")
    
    """
    out_labels = np.concatenate([np.array(["Accuracy - Current Run" for _ in range(max_len)] + ["Loss - Current Run" for _ in range(max_len)]), labels_rp])
    out_values = np.concatenate([np.array(accs[:max_len] + losses[:max_len]), values_rp])
    out_epochs = np.concatenate([np.array(epochs[:max_len] + epochs[:max_len]), epochs_rp])
    print(len(out_labels), len(out_values), len(out_epochs))
    print(out_labels)
    print(out_values)
    print(out_epochs)
    """

    plot = gr.LinePlot(value=pd.DataFrame({"Labels": np.concatenate([np.array(["Accuracy - Current Run" for _ in range(max_len)] + ["Loss - Current Run" for _ in range(max_len)]), labels_rp]), 
                                           "Values": np.concatenate([np.array(accs[:max_len] + losses[:max_len]), values_rp]), 
                                           "Epochs": np.concatenate([np.array(epochs[:max_len] + epochs[:max_len]), epochs_rp])}), 
                                           x="Epochs", y="Values", color="Labels")
    return plot

def load_graph(file_names: [str]):
    global labels_rp, epochs_rp, values_rp
    labels_rp, epochs_rp, values_rp = [], [], []
    for file_name in file_names:
        file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        plots = np.array(file.getNode("Plot").mat())
        if plots.size == 1:
            plots = np.empty((3, 0), float)
        #print(plots)
        data_points = len(plots[0])
        basename = os.path.basename(file_name)
        labels_rp = np.append(labels_rp, np.concatenate([[f"Loss - {basename}" for _ in range(data_points)], [f"Accuracy - {basename}" for _ in range(data_points)]]))
        epochs_rp = np.append(epochs_rp, np.concatenate([plots[0], plots[0]]))
        values_rp = np.append(values_rp, np.concatenate([plots[1], plots[2]]))

        #print(plots[-1])
        #print(plots[-1][0])
        #print(plots[-1][1])
        #print(plots[-1][2])
        #max_len = min([len(plots[-1][0]), len(plots[-1][1])])
        #print(["Accuracy" for _ in range(max_len)] + ["Loss" for _ in range(max_len)])
        #print(plots[-1][2] + plots[-1][1])
        #print(plots[-1][0] + plots[-1][0])
        #plot = gr.LinePlot(value=pd.DataFrame({"Labels": np.concatenate([["Accuracy" for _ in range(max_len)], ["Loss" for _ in range(max_len)]]), "Values": np.concatenate([plots[-1][2], plots[-1][1]]), "Epochs": np.concatenate([plots[-1][0], plots[-1][0]])}), x="Epochs", y="Values", color="Labels")

    #plot = gr.LinePlot(value=pd.DataFrame({"Labels": labels_rp, "Values": values_rp, "Epochs": epochs_rp}), x="Epochs", y="Values", color="Labels")
    #global visible_plots
    #visible_plots = plot

    #return plot


#def make_example_graphs():
#    file = cv2.FileStorage("test.yml", cv2.FILE_STORAGE_WRITE)
#    file.write("graph", )
#    file.release()

"""
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

x_train=mnist_trainset.data.numpy()
x_test=mnist_testset.data.numpy()
y_train=mnist_trainset.targets.numpy()
y_test=mnist_testset.targets.numpy()

x_train=x_train.reshape(60000,784)/255.0
x_test=x_test.reshape(10000,784)/255.0

mlp = MLPClassifier(hidden_layer_sizes=(32,32))
mlp.fit(x_train, y_train)

print("Training Accuracy:", mlp.score(x_train, y_train))
print("Testing Accuracy:", mlp.score(x_test, y_test))

def predictIt(img):
    img = img.reshape(1,784)/255.0
    prediction = mlp.predict(img)[0]
    return int(prediction)
"""

def predict(path, img):
    #print(img)
    #print(img.shape)
    img = img['composite']
    #img = img.reshape((28,28))
    #print(img)
    #print(img.shape)

    new_img = []
    for x in range(len(img)):
        new_img.append([])
        for y in range(len(img[x])):
            #if img[x][y][3] == 0:
            #    new_img[x].append(255)
            #else:
            #    new_img[x].append(0)
            new_img[x].append(img[x][y][3])
    img = new_img
    
    #print(img)

    model = Adjustable_model()
    #model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    #opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    checkpoint = load_checkpoint(model, path)
    model = Adjustable_model(linear_layers = checkpoint['lin_layers'], convolutional_layers = checkpoint['conv_layers'])
    #opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #model = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()
    #print(model)

    #img.reshape((1, 28, 28, 1)).astype('float32') / 255.0
    #img = img.resize((28,28))
    img = np.array(cv2.resize(np.array(img).astype('uint8'), (28,28)))
    #print(img)
    transform = transforms.Compose([transforms.ToTensor()]) #transforms.Resize(28),
    img_tensor = transform(img).unsqueeze(0)
    #print(img_tensor.shape)
    #img_tensor = img_tensor.view(img_tensor.size(0), -1)
    #img_tensor = torch.tensor(img, dtype=float)
    #img_tensor = img_tensor.float()
    #print(img_tensor) 
    #print(img_tensor.shape)

    #img = np.array(cv2.resize(np.array(img), (28,28))) #img.reshape((1, 28, 28, 1)).astype('float32') / 255.0
    prediction = model(img_tensor).data
    #print(prediction)
    pred_out, pred_index = torch.max(prediction, 1)
    #print(pred_out, pred_index)
    return pred_index.item() #str(model(np.array(cv2.resize(np.array(img['layers'][0]), (28,28)))))

    img = np.array(cv2.resize(np.array(img), (28,28))) #img.reshape((1, 28, 28, 1)).astype('float32') / 255.0
    return model(torch.from_numpy(img)) #str(model(np.array(cv2.resize(np.array(img['layers'][0]), (28,28)))))

def aaa():
    return np.zeros((28,28))

def bbb():
    return gr.update(visible=True)

visibleee = True

with gr.Blocks() as demo:
    
    with gr.Tab("Train/Test"):
        with gr.Row():
            with gr.Column():
                with gr.Tab("Select Model"):
                    gr.Markdown("Select Model")
                    button_refresh = gr.Button(value="Refresh File Explorers")
                    select_model = gr.FileExplorer("**/*.pt", label="Select Model", file_count="single", interactive=True)
                    button_refresh.click(None, js="window.location.reload()")
                    """gr.Dropdown(label="Select Dataset")
                gr.Markdown("Select Model & Dataset")
                    gr.Markdown("Select Model & Dataset")
                    gr.Dropdown(label="Select Model")
                    gr.Dropdown(label="Dataset")
                    gr.FileExplorer("**/*.pt")"""
                with gr.Tab("Create Model"):                    
                    with gr.Tab("Beginner Model Creator"):
                        in_model_name = gr.Textbox(label="Model Name", value="unnamed")
                        in_convolutional_layers = gr.Slider(label="Convolutional Layers", value=2, minimum=0, maximum=5, step=1) 
                        in_cells_per_conv = gr.Slider(label="Cells per convolutional layer", value=32, minimum=1, maximum=128, step=1)               
                        in_linear_layers = gr.Slider(label="Linear Layers", value=1, minimum=0, maximum=5, step=1)
                        in_cells_per_lin = gr.Slider(label="Cells per linear layer", value=32, minimum=1, maximum=128, step=1)
                        button_create_model = gr.Button(value="Create Model")
                        button_create_model.click(simple_model_creator, inputs=[in_model_name, in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=None)
                        button_display = gr.Button(value="Display Model")
                        #network_plot = gr.Plot()
                        
                        network_img = gr.Image(type='filepath', value='network.png')#type="pil")
                        button_display.click(make_img, inputs = [in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=network_img)          
                        #gr.Interface(make_img, gr.Image(type="pil", value=None), "image")
                        
                    with gr.Tab("Advanced Model Creator"):
                        gr.Markdown("Only recommended to people with a good understanding of ML")
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
                        #with gr.Row():
                        button_complex_create_model = gr.Button(value="Create Model")
                        network_img = gr.Image(type='filepath', value='network.png')
                        button_complex_create_model.click(complex_model_creator, inputs=[in_model_name], outputs=network_img)
                        
                        #button_create_model.click(simple_model_creator, inputs=[in_model_name, in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=None)
                        #button_display = gr.Button(value="Display Model")
                        #output = gr.Plot()
                        #button_display.click(simple_model_drawer, inputs = [in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=output)          
                        #network_plot = gr.Plot()
                        #gr.Interface(
                        #    fn=simple_model_drawer,
                        #    inputs= [in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin],
                        #    outputs=gr.Plot())
            """with gr.Column():
                with gr.Tab("Select Model"):
                    gr.Markdown("Select Model & Dataset")
                    select_model = gr.FileExplorer("**/*.pt", label="Select Model", file_count="single")
                    gr.Dropdown(label="Select Dataset")
                with gr.Tab("Create Model"):
                    gr.Markdown("Create Model")
                    model_name = gr.Textbox(label="Model Name")
                    #gr.Dropdown(label="Model Type")
                    #gr.Slider(label="Number of Layers")
                    #gr.Slider(label="Kernel Size")
                    #gr.Button(value="Create Model")
                    in_convolutional_layers = gr.Slider(label="Convolutional Layers", value=2, minimum=0, maximum=5, step=1) 
                    in_cells_per_conv = gr.Slider(label="Cells per convolutional layer", value=32, minimum=1, maximum=128, step=1)               
                    in_linear_layers = gr.Slider(label="Linear Layers", value=1, minimum=0, maximum=5, step=1)
                    in_cells_per_lin = gr.Slider(label="Cells per linear layer", value=32, minimum=1, maximum=128, step=1)
                    button_create_model = gr.Button(value="Create Model")
                    button_create_model.click(simple_model_creator, inputs=[model_name, in_convolutional_layers, in_linear_layers, in_cells_per_conv, in_cells_per_lin], outputs=None)
                    gr.Button(value="Display Model")"""
            with gr.Column():
                with gr.Tab("Adjustable Parameters"):
                    gr.Markdown("Adjustable Parameters")
                    in_learning_rate = gr.Slider(label="Learning Rate", value=0.3, minimum=0, maximum=1, step=0.01)
                    in_batch_size = gr.Slider(label="Batch Size", value=256, minimum=0, maximum=1024, step=32)
                    in_seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=1000, step=1)
                    in_n_epochs = gr.Slider(label="Epochs/Training Steps", value=10, minimum=0, maximum=50, step=1)
                    in_loss_fn = gr.Dropdown(label="Loss Function", value="CrossEntropyLoss", choices=["CrossEntropyLoss", "NLLLoss", "MSELoss", "L1Loss"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            button_start = gr.Button(value="Start/Continue")
                            #button_start.click(new_resume_training, inputs=[select_model, in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_loss_fn], outputs=None)
                            #button_start = gr.Button(value="Start")
                            #button_start.click(start_training, inputs=[in_seed, in_learning_rate, in_batch_size, in_n_epochs], outputs=None)
                        with gr.Column(min_width=100):
                            button_stop = gr.Button(value="Stop")
                            #button_stop.click(stop_training, inputs=None, outputs=None)
                        #with gr.Column(min_width=100):
                        #    button_stop = gr.Button(value="Resume")
                        #    button_stop.click(resume_training, inputs=[in_seed, in_learning_rate, in_batch_size, in_n_epochs], outputs=None)
                    with gr.Row():
                        button_revert = gr.Button(value="Revert to last epoch")
                        button_revert.click(revert_to_last_epoch, inputs=None, outputs=None)
                    with gr.Row():
                        text_component = gr.Markdown()
                    with gr.Row():
                        with gr.Column(min_width=50):
                            pass
                        with gr.Column(min_width=50):
                            spinner = gr.Image("https://csi.itpvoip.com/secure/images/new-progress-bar.gif", label="Training...", visible=False, scale=1, show_download_button=False)
                        with gr.Column(min_width=50):
                            pass
            
            button_start.click(bbb, inputs=None, outputs=spinner)
            button_start.click(new_resume_training, inputs=[select_model, in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_loss_fn], outputs=spinner)
            button_stop.click(stop_training, inputs=None, outputs=None)
                    
            with gr.Column():
                with gr.Tab("Training"):
                    gr.Markdown("Training")
                    training_plot = gr.LinePlot()
                    training_info = gr.Markdown()
                    #out_accuracy = gr.Textbox(label="Accuracy")
                    #out_loss = gr.Textbox(label="Loss")
                    select_plot = gr.FileExplorer("**/*.yml", file_count="multiple")
                    select_plot.change(load_graph, inputs=[select_plot], outputs=[])
                    #select_plot = gr.Dropdown([], value=[], multiselect=True, label="Models to plot", info="Select model training graphs to display in the plot")
                with gr.Tab("Testing"):
                    gr.Markdown("Testing")
                    #playground_in = gr.Sketchpad(crop_size=("1:1"), image_mode='L', type="numpy", interactive=True) 
                    #playground_in = gr.Sketchpad(image_mode='L', invert_colors=True, source='canvas', type="numpy") #shape = (28, 28), crop_size="1:1", 
                    #playground_in = gr.Paint(crop_size=("1:1"), image_mode='L', type="numpy", interactive=True)
                    buttoton = gr.Button(value="Magically fix Sketchpad")
                    #playground_in = gr.ImageEditor(value={'layers': [np.zeros((28,28))], 'background': None, 'composite': None}, image_mode='L', type="numpy", interactive=True)
                    playground_in = gr.Sketchpad(value=np.zeros((28,28)), crop_size=("1:1"), type="numpy", interactive=True)
                    button_test = gr.Button(value="Test")
                    playground_out = gr.Text(label="Result")
                    button_test.click(predict, inputs=[select_model, playground_in], outputs=[playground_out])
                    buttoton.click(aaa, inputs=None, outputs=[playground_in])

                """
                with gr.Tab("Playground"):
                    gr.Interface(fn=predictIt, 
                                 inputs = gr.Sketchpad(shape = (28, 28), image_mode='L', invert_colors=True, source='canvas'), 
                                 outputs = "label")
                                 """
    
    with gr.Tab("Info"):
        gr.Markdown(
"""
Introduction to Machine Learning\n\n
In order to explain the term machine learning, we must first deal with the term artificial intelligence. Artificial intelligence is a scientific discipline that focuses on the research and algorithmization of preferably human intelligence in the form of automatically usable perception and "mind power".\n\n
"Artificial intelligence is the study of computational methods that make it possible to perceive, reason and act."\n\n
Artificial intelligence (AI for short) is therefore a machine that can replicate the cognitive abilities of a human being, i.e. automates human intelligence. Philosophers and psychologists have been discussing what exactly intelligence is for thousands of years, but the ability to learn is a generally recognized component.\n
This brings us to the next term, "machine learning". Just as a person only becomes intelligent through lifelong learning, a machine only becomes intelligent through learning.\n\n
"[Machine] learning is the construction of computer programs that automatically improve through experience"\n\n
The advantage of using learning processes is that machines learn independently how best to solve certain problems. This is particularly advantageous if the problem cannot be described in concrete terms or can vary so much that there is no clearly definable solution.\n
Trainable programs are often implemented in the form of neural networks. They are modeled on the human brain, which also consists of neurons. Neural networks can be thought of as networks of neurons, i.e. information points, into which an input for a problem can be entered, which is then processed by the neurons and which then outputs a result that is closer to the desired result than the input or ideally corresponds exactly to the desired result. If the result was good, the neurons are calibrated so that they deliver more similar results. If the result was poor, they are calibrated so that they produce different results.\n
Training data is essential for such training. This includes various cases of the problem to be solved in a form that the computer can understand. There must also be a way of validating the results so that the neurons can be calibrated correctly.\n
In the case of image-to-image processing, there is one image that is processed and one that corresponds to the desired result. The image generated by the neural network from the input is then compared with the reference image.\n
The deviation between the result and the reference image is mathematically recorded in a value known as a "loss". The neural network calibrates its neurons depending on the amount of the loss, so that large changes are made if the loss is large, i.e. the generated result deviates greatly from the reference image, and small changes are made if the loss is small, i.e. the generated result is similar to the reference image. In this way, the neurons are calibrated in the long term so that the neural network achieves better and better results.\n
"""
)
    # def progress(x, progress = gr.Progress()):
    #     for epoch in range(10):
    #         progress(0, desc = "test...")
    #         time.sleep(0.1)
    #     return x
    
    # Add the row to the existing column
    # iface.add(text_component)
        
    #playground_in.value = np.zeros((2800,2800))

    # # Launch the interface
    # iface.launch()
    #demo.load(get_accuracy, None, out_accuracy, every=1)
    #demo.load(get_loss, None, out_loss, every=1)

    dep1 = demo.load(get_statistics, None, training_info, every=0.5)
    dep2 = demo.load(make_plot, None, training_plot, every=0.5)
    dep3 = demo.load(get_text, None, text_component, every=0.5)
    #dep4 = demo.load(lambda :gr.update(visible=True), None, select_model, every=0.5)
    # dep2 = demo.load(make_accuracy_plot, None, training_plot, every=0.5)
    # dep3 = demo.load(make_loss_plot, None, training_plot, every=0.5)
    
    #dep3 = demo.load(simple_model_drawer, None, network_plot)
    #demo.load(listener, None, None, every=1)
    #dep1 = demo.load(get_accuracy, None, None, every=0.5)
    #dep2 = demo.load(get_loss, None, None, every=0.5)
    #dep3 = demo.load(listener, None, None, every=0.5)

    #button_stop.click(None, None, None, cancels=[dep1, dep2])
    
    #period.change(get_accuracy_once, None, None, every=0.5, cancels=[dep])
    #dep = demo.load(get_plot, None, plot, every=0.5)
    #period.change(get_plot, period, plot, every=0.5, cancels=[dep])

if __name__ == "__main__":
    webbrowser.open_new_tab(f"http://127.0.0.1:7860/")
    demo.queue().launch()