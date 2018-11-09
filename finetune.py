import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
import time
import copy

class LandmarksDataset_Train(Dataset):
    """Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 1])
        image = io.imread(img_name + '.jpg')
        label = self.landmarks_frame.iloc[idx, 3]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label']

class LandmarksDataset_Test(Dataset):
    """Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 1])
        image = io.imread(img_name + '.jpg')
        label = self.landmarks_frame.iloc[idx, 0]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label']

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img,
                'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image /= 255
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(label))}


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, train_size):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for index, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            print('Batch_{} Loss: {:.4f} Acc: {:.4f}'.format(
            index, loss.item(), torch.sum(preds == labels.data).double() / 256))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def model_predict(model, test_loader):
    submit_txt = open('submission.txt', 'w')
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            submit_txt.write('{}\n'.format(predicted[0].item()))

    submit_txt.close()

if __name__ == '__main__':
	train_dataset = LandmarksDataset_Train(csv_file='train.csv',
                                           root_dir='./images',
                                           transform=transforms.Compose([
                                            Rescale(224),
                                            ToTensor()]))


	test_dataset = LandmarksDataset_Test(csv_file='test.csv',
	                                           root_dir='./images',
	                                           transform=transforms.Compose([
	                                               Rescale(224),
	                                               ToTensor()]))


	dataloaders = {'train': DataLoader(train_dataset, batch_size=256,
	                                             shuffle=True, num_workers=4),
	                'test': DataLoader(test_dataset, batch_size=1,
	                                             shuffle=False, num_workers=4)}

	train_size = len(train_dataset)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 10)

	model_ft = model_ft.to(device)

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	######################################################################
	# Train and evaluate
	# ^^^^^^^^^^^^^^^^^^
	#
	# It should take around 15-25 min on CPU. On GPU though, it takes less than a
	# minute.
	#

	model_ft = train_model(model_ft, dataloaders['train'], criterion, optimizer_ft, exp_lr_scheduler,
	                       5, train_size)

	model_predict(model_ft, dataloaders['test'])
































