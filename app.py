import threading
import queue
import webbrowser
import base64
import time

from io import BytesIO
from matplotlib.figure import Figure

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import torch.nn as nn
from torch import manual_seed, Tensor
from torch.optim import Optimizer, SGD
import matplotlib.pyplot as plt

from ml_utils.model import ConvolutionalNeuralNetwork
from ml_utils.training import training, load_checkpoint

app = Flask(__name__)
socketio = SocketIO(app)


# Initialize variables
seed = 42
acc = -1
loss = 0.1
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

def listener():
    global q_acc, q_loss, q_stop_signal, q_break_signal, q_epoch, \
    epoch, acc, loss, stop_signal, epoch_losses, loss_img_url
    while True:
        acc = q_acc.get()
        loss = q_loss.get()
        epoch = q_epoch.get()
        # q_epoch.put(epoch)
        while((epoch_losses.get(epoch) is None) & (epoch != -1)):
            epoch_losses[epoch] = loss
        loss_img_url = loss_plot_url()
        # q_stop_signal.put(False)
        q_acc.task_done()
        q_loss.task_done()
        q_epoch.task_done()
        # q_break_signal.task_done()
        # q_stop_signal.task_done()



@app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, epoch, epoch_losses, loss_img_url, lr, n_epochs, batch_size, loss_fn
    # render "index.html" as long as user is at "/"
    return render_template("index.html", seed=seed, acc=acc, \
                           loss=loss, epoch = epoch, loss_plot = loss_img_url, 
                           lr=lr, n_epochs=n_epochs, batch_size=batch_size,
                           loss_fn=loss_fn)

@app.route("/start_training", methods=["POST"])
def start_training():
    # ensure that these variables are the same as those outside this method
    global q_acc, q_loss, seed, stop_signal, q_stop_signal, q_break_signal, epoch, epoch_losses, loss, lr, n_epochs, batch_size, loss_fn
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    # q_stop_signal.put(False)
    print("Starting training with:")
    print(f"Seed: {seed}")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Loss function: {loss_fn}")
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
             stop_signal= stop_signal,
             q_stop_signal=q_stop_signal)
    return jsonify({"success": True})

@app.route("/stop_training", methods=["POST"])
def stop_training():
    global break_signal
    if not break_signal:
        q_stop_signal.put(True)
        # set block to true to wait for item if the queue is empty
        break_signal = q_break_signal.get(block=True)
    if break_signal:
        print("Training breaks!")
    return jsonify({"success": True})

@app.route("/resume_training", methods=["POST"])
def resume_training():
    global break_signal, epoch, lr, q_acc, q_loss, q_epoch, q_stop_signal, n_epochs, batch_size
    # q_stop_signal.put(False)
    break_signal = False
    # print(f"before get, epoch is {epoch}")
    # epoch = q_epoch.get()
    print(f"Resume from epoch {epoch}")
    path = f"stop{epoch}.pt"
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    checkpoint = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Epoch {epoch} loaded, ready to resume training!")
    training(model=model,
             optimizer=opt,
             loss_fn=loss_fn,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=checkpoint['epoch']+1,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_break_signal = q_break_signal,
             q_stop_signal=q_stop_signal)
    return jsonify({"success": True})

@app.route("/revert_to_last_epoch", methods=["GET", "POST"])
def revert_to_last_epoch():
    global break_signal, epoch, epoch_losses, loss, lr, q_epoch, loss_img_url
    # check if the training is already stopped, if not, stop first
    if not break_signal:
        q_stop_signal.put(True)
        break_signal = q_break_signal.get(block=True)
        if break_signal:
            print("Training breaks!")
    # if (q_break_signal.get(block=True)):
        # print(f"after revert epoch is {epoch}")
        # for i in range(epoch+1, n_epochs):
        #     while epoch_losses.get(i) is not None:
        #         epoch_losses[i] = None
    # to roll back to the epoch fully means forgetting everthing coming after it
    time.sleep(10)
    q_epoch.put(epoch-1) # put to self
    q_loss.put(epoch_losses[epoch-1]) # put to listener
    loss = q_loss.get()
    epoch = q_epoch.get() # get from self
    # q_epoch.put(epoch) # put to listener
    for i in range(epoch+1, n_epochs):
        while epoch_losses.get(i) is not None:
            epoch_losses[i] = None
    print(f"After revert epoch is {epoch}")
    print(f"current epoch_losses:{epoch_losses}")
    # call loss_plot to draw the new plot
    loss_img_url = loss_plot_url()
    return jsonify({"epoch_losses": epoch_losses})

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
    

@app.route("/loss_plot", methods=["GET"])
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

# @app.route("/acc_plot", methods=["GET"])
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

@app.route("/update_seed", methods=["POST"])
def update_seed():
    global seed
    seed = int(request.form["seed"])
    return jsonify({"seed": seed})

#adjust learning rate 
@app.route("/update_learningRate", methods=["POST"])
def update_learningRate():
    global lr
    lr = float(request.form["lr"])
    return jsonify({"lr": lr})

#adjust number of epochs
@app.route("/update_numEpochs", methods=["POST"])
def update_numEpochs():
    global n_epochs
    n_epochs = int(request.form["n_epochs"])
    return jsonify({"n_epochs": n_epochs})

#adjust batch_size
@app.route("/update_batch_size", methods=["POST"])
def update_batch_size():
    global batch_size
    batch_size = int(request.form["batch_size"])
    return jsonify({"batch_size": batch_size})

#adjust loss_fn
@app.route("/update_loss_fn", methods=["POST"])
def update_loss_fn():
    global loss_fn
    loss_dict = {"CrossEntropy":nn.CrossEntropyLoss(),
                 "NegativeLog":nn.NLLLoss(),
                 "L1":nn.L1Loss(),
                 "MSE":nn.MSELoss()}
    loss_fn = loss_dict[str(request.form["loss_fn"])]
    return jsonify({"success": True})

@app.route("/get_accuracy")
def get_accuracy():
    global acc
    return jsonify({"acc": acc})

@app.route("/get_loss")
def get_loss():
    global loss
    return jsonify({"loss": loss})

@app.route("/get_epoch")
def get_epoch():
    global epoch
    return jsonify({"epoch": epoch})

@app.route("/get_epoch_losses")
def get_epoch_losses():
    global epoch_losses
    return jsonify({"epoch_losses": epoch_losses})

@app.route("/get_dict")
def get_dict():
    dictTest = dict({"one": "1", "two": "2"})
    return jsonify({"dictTest": dictTest})

@app.route("/get_loss_image")
def get_loss_image():
    global loss_img_url
    return jsonify({"loss_img_url": loss_img_url})

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5001
    print("App started")
    threading.Thread(target=listener, daemon=True).start()
    webbrowser.open_new_tab(f"http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=True)
