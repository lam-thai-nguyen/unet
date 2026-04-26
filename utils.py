import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):

    history_train = []
    history_test = []
    for i in range(epochs):
        total_loss_train = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        total_loss_test, total_dice = evaluate(test_dataloader, model, loss_fn)
        history_train.append(total_loss_train)
        history_test.append(total_loss_test)
        if i % 1 == 0:
            print(f"Epoch {i}: Train loss {total_loss_train:4f} Test loss: {total_loss_test:4f} Dice score: {total_dice:4f}")

    return history_train, history_test


def train_one_epoch(dataloader, model, loss_fn, optimizer):

    model.train()
    total_loss = 0.

    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        loss = loss_fn(pred,label)
        total_loss += loss.item()

        # L1 regularization
        l1_lambda = 1e-5
        l1_reg = sum(p.abs().sum() for p in model.parameters())
        l1_loss = l1_lambda * l1_reg

        # L2 regularization
        l2_lambda = 1e-4
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        l2_loss = l2_lambda * l2_reg

        # Combined loss
        loss = loss + l1_loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_batch = len(dataloader)

    return total_loss / num_batch


def evaluate(dataloader, model, loss_fn):

    model.eval()
    total_loss = 0.
    total_dice = 0.

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_fn(pred,label)
            total_loss += loss.item()
            dice = dice_score(pred, label)
            total_dice += dice.item()

    num_batch = len(dataloader)

    return total_loss / num_batch, total_dice / num_batch


def dice_score(logits, target, smooth=1e-6):
    """logits (1,2,512,512) target(1,1,512,512)"""
    probs = logits.softmax(dim=1)[:, 1]
    pred = (probs > 0.5).float()
    target = target.squeeze(1).float()
    
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def show_img_pair(img1, img2):
    """show img1 and img2 side by side"""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    # case 1: inputs are paths
    if isinstance(img1, str) and isinstance(img1, str):
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        ax1.imshow(img1, cmap='gray')
        ax2.imshow(img2, cmap='gray')

    # case 2: inputs are Tensors
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        ax1.imshow(img1.squeeze(0), cmap='gray')
        ax2.imshow(img2.squeeze(0), cmap='gray')

    # case 3: inputs are ndarrays
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        ax1.imshow(img1, cmap='gray')
        ax2.imshow(img2, cmap='gray')

    plt.tight_layout()
    plt.show()


def show_loss(history_train, history_test):
    plt.figure(figsize=(4,4))
    plt.plot(history_train, label="train loss")
    plt.plot(history_test, label="test loss")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def pred_to_mask(pred):
    """turn pred tensor(1,2,512,512) to mask ndarray(512,512)"""
    pred = pred.cpu()
    return torch.argmax(pred.squeeze(0), dim=0).numpy()
