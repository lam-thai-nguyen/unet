import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISBI(Dataset):

    def __init__(self, root, train, transform=None):
        """
        root (str): where to look for your images and labels
        train (bool): True for training set, False for test set
        transform: specify your albumentations transformation
        """
        self.images_dir = os.path.join(root, "train/images/") if train else os.path.join(root, "test/images/")
        self.labels_dir = os.path.join(root, "train/labels/") if train else os.path.join(root, "test/labels/")
        self.images_path = [os.path.join(self.images_dir, i) for i in os.listdir(self.images_dir) if i.endswith(".png")]
        self.labels_path = [os.path.join(self.labels_dir, l) for l in os.listdir(self.labels_dir) if l.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = read_image(image_path)
        label_path = self.labels_path[index]
        label = read_image(label_path)

        if self.transform:
            augmented = self.transform(image=image.squeeze(0).numpy(), mask=label.squeeze(0).numpy())
            image = augmented["image"]
            label = augmented["mask"]

        label = (label / 255).long()

        return (image, label)
    

train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Affine(translate_percent=(-0.05, 0.05), rotate=(-90, 90), p=0.5),
    A.GridElasticDeform(num_grid_xy=(3, 3), magnitude=10, interpolation=1, mask_interpolation=0),
    A.RandomBrightnessContrast(p=1.0),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])


test_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])