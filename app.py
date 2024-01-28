import os
os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.50.2")
import queue
import webbrowser
import base64

from io import BytesIO
from matplotlib.figure import Figure

from flask import render_template, request, jsonify
import numpy as np
from sklearn.neural_network import MLPClassifier
from torch import manual_seed
from torch.optim import SGD

from ml_utils.model import Adjustable_model
from ml_utils.training import training, load_checkpoint

import torchvision.datasets as datasets

import gradio as gr
import numpy as np
import pandas as pd


# app = Flask(__name__)
# socketio = SocketIO(app)


# Initialize variables
# seed = 42
acc = -1
loss = 0.1
n_epochs = 10
epoch = -1
epoch_losses = dict.fromkeys(range(n_epochs))
stop_signal = False
data_image = base64.b64encode(b"").decode("ascii")
loss_img_url = f"data:image/png;base64,{data_image}"
lr = 0.3
batch_size = 256
q_acc = queue.Queue()
q_loss = queue.Queue()

q_stop_signal = queue.Queue()
q_epoch = queue.Queue()
q_loss_img = queue.Queue()

accs = []
losses = []
epochs = []


# Default: For now two convolutional layers (The interface will have to be adjusted to be able to set the parameters): 
conv_layer1 = {'size' : 32, 'kernel_size' : 8, 'stride' : 2, 'padding' : 2}
conv_layer2 = {'size' : 32, 'kernel_size' : 4, 'stride' : 2, 'padding' : 0}
#in_convolutional_layers = 2
conv_layers_proto =  [conv_layer1, conv_layer2, {'size' : 32}, {'size' : 32}, {'size' : 32}]

# Default: One hidden layer with 32 cells:
#lin1 = 32
#lin_layers = [32, 32, 32, 32, 32, 32] #, lin2] # Linear layers only have layer size as a parameter



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


# @app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, epoch_losses, loss_img_url, lr, n_epochs, batch_size
    # render "index.html" as long as user is at "/"
    return render_template("index.html", seed=seed, acc=acc, \
                           loss=loss, loss_plot = loss_img_url, lr=lr, n_epochs=n_epochs, batch_size=batch_size)

# @app.route("/start_training", methods=["POST"])
def start_training(seed, learning_rate, batch_size, n_epochs, lin_layer_num, conv_layer_num):
    print("starting Training with seed " + str(seed))
    # ensure that these variables are the same as those outside this method
    global q_acc, q_loss, stop_signal, epoch, epoch_losses, loss
    
    lin_layers = [32 for i in range(lin_layer_num)]
    conv_layers = [conv_layers_proto[i] for i in range(conv_layer_num)]
    
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #print(seed)
    #print(learning_rate)
    #print(n_epochs)
    #print(batch_size)
    # execute training
    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=0,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_stop_signal=q_stop_signal)
    return #jsonify({"success": True})

# @app.route("/stop_training", methods=["POST"])
def stop_training():
    global stop_signal, q_stop_signal
    if q_stop_signal is not None:
        q_stop_signal.put(True)
    stop_signal = True  # Set the stop signal to True
    # saveCheckpoint()
    return #jsonify({"success": True})

# @app.route("/resume_training", methods=["POST"])
def resume_training(seed, learning_rate, batch_size, n_epochs, lin_layer_num, conv_layer_num):
    global stop_signal
    
    lin_layers = [32 for i in range(lin_layer_num)]
    conv_layers = [conv_layers_proto[i] for i in range(conv_layer_num)]

    manual_seed(seed)
    np.random.seed(seed)

    path = "stop.pt"
    if q_stop_signal is not None:
        q_stop_signal.put(False)
    stop_signal = False  # Set the stop signal to False
    model = Adjustable_model(linear_layers = lin_layers, convolutional_layers = conv_layers)
    opt = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    # checkpoint = torch.load(PATH)
    checkpoint = load_checkpoint(model, path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=n_epochs,
             start_epoch=checkpoint['epoch']+1,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss,
             q_epoch=q_epoch,
             q_stop_signal=q_stop_signal)
    return #jsonify({"success": True})

# @app.route("/loss_plot", methods=["GET"])
# loss_plot is for the display at endpoint /loss_plot while loss_plot_2 is for the display at index.html
def loss_plot():
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

# @app.route("/get_accuracy")
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

# @app.route("/get_loss_image")
def get_loss_image():
    global loss_img_url
    return jsonify({"loss_img_url": loss_img_url})

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
    Epoch: \t {epoch}\n
    Accuracy: \t {acc}\n
    Loss: \t {loss}
"""
#str("Epoch:         " + str(epoch) + "\n" + "Accuracy:      " + str(acc) + "\n" + "Loss:          " + str(loss))

def make_plot():
    global accs, losses
    training_steps = []
    max_len = min([len(accs), len(losses)])
    for j in range(2):
        for i in range(max_len):
            training_steps.append(i + 1)
    #plot = gr.LinePlot(value=pd.DataFrame({"Epoch": training_steps, "Accuracy": accs, "Loss": losses}), x="Epoch", y="Accuracy")
    plot = gr.LinePlot(value=pd.DataFrame({"Labels": ["Accuracy" for _ in range(max_len)] + ["Loss" for _ in range(max_len)], "Values": accs[:max_len] + losses[:max_len], "Training Steps": training_steps}), x="Training Steps", y="Values", color="Labels")
    return plot

    
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

with gr.Blocks() as demo:
    
    with gr.Tab("Train/Test"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Select Model & Dataset")
                gr.Dropdown(label="Select Model")
                gr.Dropdown(label="Dataset")
                gr.FileExplorer("**/*.ckpt")
                gr.Markdown("Create Model")
                gr.Dropdown(label="Model Type")
                in_convolutional_layers = gr.Slider(label="Convolutional Layers", value=2, minimum=0, maximum=5, step=1)                
                in_linear_layers = gr.Slider(label="Linear Layers", value=1, minimum=0, maximum=5, step=1)
                gr.Button(value="Display Model")
            with gr.Column():
                gr.Markdown("Adjustable Parameters")
                in_learning_rate = gr.Slider(label="Learning Rate", value=0.3, minimum=0, maximum=1, step=0.01)
                in_batch_size = gr.Slider(label="Batch Size", value=256, minimum=0, maximum=1024, step=32)
                in_seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=1000, step=1)
                in_n_epochs = gr.Slider(label="Epochs/Training Steps", value=10, minimum=0, maximum=100, step=1)
                gr.Dropdown(label="Loss Function")
                with gr.Row():
                    with gr.Column(min_width=100):
                        button_start = gr.Button(value="Start")
                        button_start.click(start_training, inputs=[in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_linear_layers, in_convolutional_layers], outputs=None)
                    with gr.Column(min_width=100):
                        button_stop = gr.Button(value="Stop")
                        button_stop.click(stop_training, inputs=None, outputs=None)
                    with gr.Column(min_width=100):
                        button_continue = gr.Button(value="Continue")
                        button_continue.click(resume_training, inputs=[in_seed, in_learning_rate, in_batch_size, in_n_epochs, in_linear_layers, in_convolutional_layers], outputs=None)
            with gr.Column():
                with gr.Tab("Training"):
                    gr.Markdown("Training")
                    training_plot = gr.LinePlot()
                    #out_accuracy = gr.Textbox(label="Accuracy")
                    #out_loss = gr.Textbox(label="Loss")
                    training_info = gr.Markdown()
                    gr.Markdown("Analysis")
                    gr.Markdown("...")
                with gr.Tab("Testing"):
                    gr.Markdown("Test Result")
                    gr.Image(label="Input")
                    gr.Button(value="Select random Image")
                    gr.Image(label="Output")
                    gr.Markdown("Label: ...")
    
    with gr.Tab("Playground"):
        gr.Interface(fn=predictIt, 
                     inputs = gr.Sketchpad(shape = (28, 28), image_mode='L', invert_colors=True, source='canvas'), 
                     outputs = "label")

    
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

    #demo.load(get_accuracy, None, out_accuracy, every=1)
    #demo.load(get_loss, None, out_loss, every=1)
    dep1 = demo.load(get_statistics, None, training_info, every=0.5)
    dep2 = demo.load(make_plot, None, training_plot, every=0.5)
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