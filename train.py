import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet, ISBI, train_transform, test_transform
from torch.utils.data import DataLoader
from utils import train, show_img_pair, show_loss, pred_to_mask 



if __name__ == "__main__":
    # Preparation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    model = UNet().to(device)
    training_data = ISBI(root="./data", train=True, transform=train_transform)
    test_data = ISBI(root="./data", train=False, transform=test_transform)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=3e-3)
    # Train
    history_train, history_test = train(train_dataloader, test_dataloader, model, criterion, optimizer, 1)
    show_loss(history_train, history_test)
    # Visualize train
    img, label = training_data[0]
    img = img.to(device).unsqueeze(0)
    label = label.to(device)
    pred = model(img)
    mask = pred_to_mask(pred)
    show_img_pair(label.cpu().squeeze(0).numpy(), mask)  # gt vs pred
    # Visualize test
    img, label = test_data[10]
    img = img.to(device).unsqueeze(0)
    label = label.to(device)
    pred = model(img)
    mask = pred_to_mask(pred)
    show_img_pair(label.cpu().squeeze(0).numpy(), mask)  # gt vs pred
