Implementation of 'U-Net: Convolutional Networks for Biomedical Image Segmentation'

Paper: https://arxiv.org/abs/1505.04597

***

Scope:
- Implement U-Net in PyTorch
- Implement data augmentation (like the paper)
- Train and evaluate U-Net on ISBI dataset using Dice score

File navigation:
- unet/model.py: U-Net model (the final product)
- unet/dataset.py: ISBI class, train_transform, test_transform
- train.py: you only need to run this file to train the model and view inference
- utils.py: includes the components that are used in train.py 
- unet.ipynb: The thinking process (lengthy but informative)
(View unet.ipynb in Kaggle: https://www.kaggle.com/code/thaimeuu/unet-implementation)
(View unet.ipynb in NBViewer: https://nbviewer.org/github/lam-thai-nguyen/unet/blob/main/unet.ipynb)

My recommendation:
- View all .py files to see the structure (no understanding needed)
- Then view all the images and their caption (no understanding needed)
- If interested, view unet.ipynb carefully (if you understand this file, you don't have to go back to understand *.py files)

View output images:
- Go to ./images/
- Refer to the image file name for its context