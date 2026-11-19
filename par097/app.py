from huggingface_hub import hf_hub_download   # downloads a file from the Hugging Face Hub (already installed with gradio)
import json                                  # read class_names.json
import numpy as np                           # arrays, used to build the heatmap picture
import torch                                 # main deep learning library
import torch.nn as nn                        # layers (we need nn.Linear)
import torch.nn.functional as F              # softmax, relu, interpolate
import gradio as gr                          # builds the web page
from PIL import Image                        # open and handle images
from torchvision import models, transforms   # ResNet architecture and image preprocessing
from matplotlib import colormaps             # color scale for the heatmap

device = "cpu"                               # the app runs on the CPU, which is fast enough for one image

REPO = "santaparvin/brain-mri-resnet18"                    # your model repository on Hugging Face
names_path = hf_hub_download(REPO, "class_names.json")     # downloads the file once and keeps a local copy
weights_path = hf_hub_download(REPO, "model.pth")          # same for the trained weights
with open(names_path) as f:                                # open the downloaded class names
    class_names = json.load(f)                             # list: glioma, meningioma, notumor, pituitary

model = models.resnet18(weights=None)                                  # ResNet-18 structure only, no download needed
model.fc = nn.Linear(model.fc.in_features, len(class_names))           # same final layer as in training (4 classes)
model.load_state_dict(torch.load(weights_path, map_location=device))  # load YOUR trained weights from the downloaded file
model.eval()                                                           # evaluation mode: no augmentation/dropout behavior

tf = transforms.Compose([                                              # same preprocessing as eval_tf in train.py
    transforms.Resize((224, 224)),                                     # ResNet expects 224 x 224
    transforms.ToTensor(),                                             # image to tensor, values 0 to 1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # same normalization as in training
])

feats, grads = {}, {}                        # boxes to store values captured during the forward/backward pass


def save_grad(g):                            # called automatically during backpropagation
    grads["v"] = g                           # store the gradient of the last conv features


def fwd_hook(module, inp, out):              # called automatically after layer4 runs
    feats["v"] = out                         # store the feature maps of the last conv block
    out.register_hook(save_grad)             # ask PyTorch to also store their gradient later


model.layer4.register_forward_hook(fwd_hook)   # attach the hook to ResNet's last convolutional block


def predict(image):
    if image is None:                        # nothing uploaded yet
        return None, None
    image = image.convert("RGB")             # make sure the image has 3 color channels
    x = tf(image).unsqueeze(0)               # preprocess and add a batch dimension: (1, 3, 224, 224)
    model.zero_grad()                        # clear old gradients
    out = model(x)                           # forward pass: one score for each class
    probs = F.softmax(out, dim=1)[0]         # convert scores to probabilities that sum to 1        # convert scores to probabilities that sum to 1
    pred = int(probs.argmax())               # index of the most likely class

    out[0, pred].backward()                  # backpropagate the winning class score (needed for Grad-CAM)
    weights = grads["v"].mean(dim=(2, 3), keepdim=True)          # importance of each feature map
    cam = F.relu((weights * feats["v"]).sum(dim=1, keepdim=True))  # weighted sum, keep only positive evidence
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]  # enlarge to image size
    cam = cam - cam.min()                    # shift so the smallest value is 0
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()            # scale to 0..1 and convert to numpy

    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)   # color the heatmap (blue = low, red = high)
    base = np.array(image.resize((224, 224)))                    # original image at the same size
    overlay = (0.55 * base + 0.45 * heat).astype(np.uint8)       # blend the image and the heatmap

    labels = {class_names[i]: float(probs[i]) for i in range(len(class_names))}  # class name -> probability
    return labels, Image.fromarray(overlay)  # results shown on the web page


demo = gr.Interface(                         # builds the web interface
    fn=predict,                              # function to run when an image is uploaded
    inputs=gr.Image(type="pil", label="Upload a brain MRI image"),   # input box
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction"),             # shows the probability of each class
        gr.Image(label="Grad-CAM heatmap (where the model looked)"), # shows the heatmap overlay
    ],
    title="Brain Tumor MRI Classifier",
    description="Educational demo only. This is NOT a medical diagnosis. "
                "Classes: glioma, meningioma, pituitary tumor, no tumor.",
)

if __name__ == "__main__":                   # start the app only when this file is run directly
    demo.launch()                            # starts a local web server