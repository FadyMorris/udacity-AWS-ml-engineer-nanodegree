#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import logging
import sys
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time





def test(model, test_loader, criterion, device, batch_size):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing started!")
    model.eval()
    running_losses=[]
    running_corrects=0

    with torch.no_grad():    
        for inputs, labels in test_loader:
            # GPU Training              
            inputs=inputs.to(device)
            labels=labels.to(device)

            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_losses.append(loss.item())
            running_corrects += torch.sum(preds == labels.data)

    total_loss = sum(running_losses) / len(running_losses)
    total_acc = torch.div(running_corrects.double(), len(test_loader)).item()
    total_acc = running_corrects.double().item() / len(test_loader.dataset)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, device, batch_size, hook=None):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    logger.info("Training started!")
    epochs=2 #only two epochs for tuning job
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            dataset_length = len(image_dataset[phase].dataset)
            if phase=='train':
                model.train()
                grad_enabled = True

            else:
                model.eval()
                grad_enabled = False

            running_losses = []
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                # GPU Training              
                inputs=inputs.to(device)
                labels=labels.to(device)

                with torch.set_grad_enabled(grad_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

                running_losses.append(loss.item())
                running_corrects += torch.sum(preds == labels.data)
                processed_images_count = batch_idx * batch_size + len(inputs)
                logger.info(
                        "{} epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                phase,
                                epoch,
                                processed_images_count,
                                dataset_length,
                                100.0 * processed_images_count / dataset_length,
                                loss.item(),
                            )
                           )
                          
            epoch_loss = sum(running_losses) / len(running_losses)
            epoch_acc = running_corrects.double().item() / len(image_dataset[phase].dataset)

            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1  #Early Stopping if validation loss gets worse.

            status = '{} loss: {:.4f}, acc: {:.4f}, best validation loss: {:.4f}'.format(phase,epoch_loss,epoch_acc,best_loss)
            logger.info(status)
            
        if loss_counter==1: #Early Stopping
            break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    output_size = 133
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, output_size))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()








    # Multiple GPU Tranining (Data Parallelism)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multiple_gpus_exist = torch.cuda.device_count() > 1
    if multiple_gpus_exist:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    logger.info(f"Transferring Model to Device {device}")
    model=model.to(device)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(
                          model.module.fc.parameters() if multiple_gpus_exist else model.fc.parameters(), 
                          lr=args.learning_rate
                          )

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)



    start_time = time.time()
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.batch_size)
    logger.info("Training time: {} seconds.".format(round(time.time() - start_time, 2)))

    '''
    TODO: Test the model to see its accuracy
    '''
    start_time = time.time()
    test(model, test_loader, loss_criterion, device, args.batch_size)
    logger.info("Testing time: {} seconds.".format(round(time.time() - start_time, 2)))

    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING']) # --data-dir
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
