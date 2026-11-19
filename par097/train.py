import argparse                                   # read settings (like the data folder) from the command line
import json                                       # save the class names for the app later
import torch                                      # the main deep learning library
import torch.nn as nn                             # layers and loss functions
from torch.utils.data import DataLoader, Subset   # feed data in batches / take part of a dataset
from torchvision import datasets, transforms, models  # image folders, preprocessing, pretrained models
from sklearn.metrics import classification_report, confusion_matrix  # evaluation metrics


def get_args():
    p = argparse.ArgumentParser()                                # object that reads command-line options
    p.add_argument("--data_dir", required=True)                  # folder that contains Training/ and Testing/
    p.add_argument("--epochs", type=int, default=5)              # how many passes over the training images
    p.add_argument("--batch_size", type=int, default=32)         # how many images are processed at once
    p.add_argument("--lr", type=float, default=1e-4)             # learning rate: small, because we fine-tune
    return p.parse_args()                                        # give back the values


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None                  # an optimizer is given only when we are training
    model.train(training)                             # train mode when training, eval mode otherwise
    total_loss, correct, total = 0.0, 0, 0            # running totals for this epoch
    with torch.set_grad_enabled(training):            # calculate gradients only when training (saves memory)
        for x, y in loader:                           # x = batch of images, y = their true labels
            x, y = x.to(device), y.to(device)         # move the data to the GPU if there is one
            out = model(x)                            # forward pass: a score for each class
            loss = criterion(out, y)                  # how wrong the predictions are
            if training:
                optimizer.zero_grad()                 # clear old gradients, otherwise they add up
                loss.backward()                       # backpropagation: find how to improve the weights
                optimizer.step()                      # update the weights
            total_loss += loss.item() * x.size(0)     # add this batch's loss, weighted by batch size
            correct += (out.argmax(1) == y).sum().item()  # count correct predictions (highest score wins)
            total += x.size(0)                        # count images seen
    return total_loss / total, correct / total        # average loss and accuracy


def main():
    args = get_args()                                              # read the options
    device = "cuda" if torch.cuda.is_available() else "cpu"        # use the GPU if available, otherwise the CPU
    print("Using device:", device)                                 # so you know which one is used

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]       # ImageNet statistics the pretrained model expects
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),                             # ResNet expects 224 x 224 images
        transforms.RandomHorizontalFlip(),                         # augmentation: more variety, less overfitting
        transforms.RandomRotation(10),                             # augmentation: small random rotations
        transforms.ToTensor(),                                     # image to tensor, values 0 to 1
        transforms.Normalize(mean, std),                           # normalize like the pretrained model's data
    ])
    eval_tf = transforms.Compose([                                 # no augmentation when evaluating
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = f"{args.data_dir}/Training"                        # training images folder
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)   # the label comes from the subfolder name
    val_full = datasets.ImageFolder(train_dir, transform=eval_tf)      # same images, but without augmentation
    n = len(train_full)                                            # number of training images
    n_val = int(0.15 * n)                                          # keep 15% for validation
    g = torch.Generator().manual_seed(42)                          # fixed seed, so the split is repeatable
    idx = torch.randperm(n, generator=g).tolist()                  # shuffled image numbers
    train_ds = Subset(train_full, idx[n_val:])                     # 85% for training
    val_ds = Subset(val_full, idx[:n_val])                         # 15% for validation (choose the best epoch)
    test_ds = datasets.ImageFolder(f"{args.data_dir}/Testing", transform=eval_tf)  # untouched final test set

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)   # shuffle every epoch
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)    # num_workers=0 avoids Windows errors
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    class_names = train_full.classes                               # glioma, meningioma, notumor, pituitary
    print("Classes:", class_names)                                 # check that 4 classes were found
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)                                  # the app needs the same label order

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # ResNet-18 pretrained on ImageNet
    model.fc = nn.Linear(model.fc.in_features, len(class_names))       # new last layer with our 4 classes
    model = model.to(device)                                       # move the model to the GPU if there is one

    criterion = nn.CrossEntropyLoss()                              # standard loss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   # Adam optimizer, updates all layers

    best_val = 0.0                                                 # best validation accuracy so far
    for epoch in range(args.epochs):                               # repeat for each epoch
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)  # learn from training data
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)               # check on validation data
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")        # progress, also useful for your report
        if va_acc > best_val:                                      # keep only the best model
            best_val = va_acc
            torch.save(model.state_dict(), "model.pth")            # save the weights to a file

    model.load_state_dict(torch.load("model.pth", map_location=device))   # reload the best weights
    model.eval()                                                   # evaluation mode
    preds, labels = [], []                                         # to collect all predictions
    with torch.no_grad():                                          # no gradients needed for testing
        for x, y in test_dl:
            out = model(x.to(device))                              # forward pass
            preds += out.argmax(1).cpu().tolist()                  # predicted class numbers
            labels += y.tolist()                                   # true class numbers
    print(classification_report(labels, preds, target_names=class_names, digits=3))  # precision, recall, F1
    print(confusion_matrix(labels, preds))                         # which classes get confused


if __name__ == "__main__":                                         # run main() only when this file is run directly
    main()