from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
from base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Resize
import torchvision
from tqdm import tqdm
from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    plt.ion()   # interactive mode

    data_dir = '../datas'
    BATCH_SIZE = 8

    data_transforms = transforms.Compose([
            #Preproc(0.2),
            Rescale(224),
            # RandomCrop(224),
            transforms.CenterCrop(224),
            #transforms.RandomRotation((-45, 45)),
            # Resize(224),
            ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test_data'), data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=4)

    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders))


    def test(model, criterion):
        best_model_wts = torch.load('best_model.pth')
        model.load_state_dict(best_model_wts)

        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        pred_array = []
        target_array = []
        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)



            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                pred_array.append(preds.to('cpu').numpy())
                target_array.append(labels.to('cpu').numpy())

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        loss = running_loss / dataset_sizes
        acc = running_corrects.double() / dataset_sizes

        cnf_matrix = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        
        '''
        for i in range(len(np.array(target_array).ravel())):
            result = confusion_matrix(np.array(target_array).ravel()[i], np.array(pred_array).ravel()[i])
            for j in range(len(result)):
                for k in range(len(result)):
                    cnf_matrix[j][k] = cnf_matrix[j][k] + result[j][k]
        '''

        
        for i in range (len(np.array(target_array).ravel())):
            cnf_matrix[np.array(target_array).ravel()[i]][np.array(pred_array).ravel()[i]] += 1
            


        #pred = [0, 0, 0, 0, 0, 0, 0, 0]
        #for i in range(len(np.array(pred_array).ravel())):
         #   result = np.array(pred_array).ravel()[i]
          #  for j in range(len(result)):
           #         pred[j] = pred[j] + result[j]
        
        #print(target)
        #print(pred)
        
        plot_confusion_matrix(cnf_matrix, np.array(['CNV', 'DME', 'DRUSEN', 'NORMAL']), normalize=False, title='Confusion Matrix')
        print('test Loss: {:.4f} Acc: {:.4f}'.format(
            loss, acc))

        #cnf_matrix = confusion_matrix(target, pred)
        #print(cnf_matrix)
        #cnf_matrix = cnf_matrix0 + cnf_matrix1;
        #np.set_printoptions(precision=2)

        #Plot confusion matrix
        #plt.figure()
        #plot(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')

        #plt.show()

        print()
        return model

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//4, 4, images_so_far)
                    ax.axis('off')
                    ax.set_title('label: {} / pred: {}'.format(class_names[labels[j]], class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        return


    model_ft = models.resnet34(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,4)

    model_ft = model_ft.to(device)


    criterion = nn.CrossEntropyLoss()

    model_ft = test(model_ft, criterion)


    visualize_model(model_ft, 12)
    plt.ioff()
    plt.show()


