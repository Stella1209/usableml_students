import threading
import queue
import webbrowser
import base64

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
loss_fn=None#nn.CrossEntropyLoss()
n_epochs = 10
epoch = -1
epoch_losses = dict.fromkeys(range(n_epochs))
stop_signal = False
data_image = base64.b64encode(b"").decode("ascii")
loss_img_url = f"data:image/png;base64,{data_image}"
loss_img_url = f"data:image/png;base64,{data_image}"
lr = 0.3
batch_size = 256
q_acc = queue.Queue()
q_loss = queue.Queue()

q_stop_signal = queue.Queue()
q_epoch = queue.Queue()
q_loss_img = queue.Queue()




def listener():
    global q_acc, q_loss, q_stop_signal, q_epoch, q_loss_img, \
    epoch, acc, loss, stop_signal, epoch_losses, loss_img_url
    while True:
        acc = q_acc.get()
        loss = q_loss.get()
        epoch = q_epoch.get()
        while((epoch_losses.get(epoch) is None) & (epoch != -1)):
            epoch_losses[epoch] = loss
            data_url = loss_plot_2()
            q_loss_img.put(data_url)
        loss_img_url = q_loss_img.get()
        q_stop_signal.put(stop_signal)
        q_acc.task_done()
        q_loss.task_done()
        q_epoch.task_done()
        q_stop_signal.task_done()


@app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, epoch_losses, loss_img_url, lr, n_epochs, batch_size,loss_fn
    # render "index.html" as long as user is at "/"
    return render_template("index.html", seed=seed, acc=acc, \
                           loss=loss, loss_plot = loss_img_url, lr=lr, n_epochs=n_epochs, batch_size=batch_size) #, loss_fn=loss_fn

@app.route("/start_training", methods=["POST"])
def start_training():
    # ensure that these variables are the same as those outside this method
    global q_acc, q_loss, seed, stop_signal, epoch, epoch_losses, loss, lr, n_epochs, batch_size, loss_fn
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    print(seed)
    print(lr)
    print(n_epochs)
    print(batch_size)
    print(loss_fn)
    # execute training
    training(model=model,
             optimizer=opt,
             loss_fn=loss_fn,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=0,
             batch_size=256,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_stop_signal=q_stop_signal)
    return jsonify({"success": True})

@app.route("/stop_training", methods=["POST"])
def stop_training():
    global stop_signal
    stop_signal = True  # Set the stop signal to True
    # saveCheckpoint()
    return jsonify({"success": True})

@app.route("/resume_training", methods=["POST"])
def resume_training():
    global stop_signal
    path = "stop.pt"
    stop_signal = False  # Set the stop signal to False
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=0.3, momentum=0.5)
    # checkpoint = torch.load(PATH)
    checkpoint = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
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
             q_stop_signal=q_stop_signal)
    return jsonify({"success": True})

@app.route("/loss_plot", methods=["GET"])
def loss_plot():
    global epoch_losses, loss, epoch, data_url
    # while((epoch_losses.get(epoch) is None) & (epoch != -1)):
    #     epoch_losses[epoch] = loss  
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
    data_url = f"<img src='data:image/png;base64,{data_image}'/>"
    return data_url

def loss_plot_2():
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
    epochs = int(request.form["n_epochs"])
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
                 "HingeEmbedding":nn.HingeEmbeddingLoss()}
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
