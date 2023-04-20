import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def test(model, test_loader, criterion, device, epoch_no):
    """
    Tests the given model on the complete testing dataset, and prints the average loss and accuracy.

    Parameters:
        model (torch.nn.Module): The PyTorch model to test.
        test_loader (torch.utils.data.DataLoader): The testing dataset loader.
        criterion (torch.nn.Module): The loss function to use.
        device (str): The device to use for running the model and the loss function.
        epoch_no (int): The current epoch number.

    Returns:
        None.
    """
    print(f"Epoch: {epoch_no} - Testing Model on Complete Testing Dataset")
    model.eval()
    running_loss = 0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) #calculate running loss
            running_corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate running corrects

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        print( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))

def train(model, train_loader, criterion, optimizer, device, epoch_no):
    """
    Trains the model on the complete training dataset for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.Module): The loss function to be optimized.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        device (str): The device to be used for training.
        epoch_no (int): The current epoch number.
    Returns:
        torch.nn.Module: The trained model.
    """    
    print(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset" )
    model.train()
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        running_loss += loss.item() * inputs.size(0) #calculate running loss
        running_corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate running corrects
        running_samples += len(inputs) #keep count of running samples
        loss.backward()
        optimizer.step()
        if running_samples % 500 == 0:
            print("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*(running_corrects/ running_samples)
            ))
    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects/ len(train_loader.dataset)
    print( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, running_corrects, len(train_loader.dataset), 100.0 * total_acc
    ))   
    return model
    
def net():
    """
    Returns a pretrained ResNet50 model with the final fully connected layer modified to output 133 nodes 
    for predicting dog breeds.
    
    Returns:
    - model (torchvision.models.ResNet): Pretrained ResNet50 model with modified final layer
    """
    model = models.resnet50(pretrained = True) #Use a pretrained resnet50 model with 50 layers
    
    for param in model.parameters():
        param.requires_grad = False #Freeze all the Conv layers
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 256), #Add two fully connected layers
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True) # output should have 133 nodes as we have 133 classes of dog breeds
                            )
    return model

def create_data_loaders(data, batch_size):
    """
    Create PyTorch data loaders for training and testing datasets.

    Args:
        data (str): Path to the directory containing the 'train' and 'test' sub-directories.
        batch_size (int): Batch size for training and testing data loaders.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple of PyTorch data loaders for training and testing datasets.
    """
    train_dataset_path = os.path.join(data, "train")
    test_dataset_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )
    
    return train_data_loader, test_data_loader

def main(args):
    """
    The main function that trains the model.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None.
    """
    # Determine if GPU is available, and set the device accordingly.    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model and move it to the appropriate device.
    model=net()
    model = model.to(device)
    
    # Create data loaders for the training and testing data.
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size )
    
    loss_criterion = nn.CrossEntropyLoss()
    # Instantiate an optimizer for the model's fully connected layers.
   
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)

    # Train and test the model for the specified number of epochs.
    for epoch_no in range(1, args.epochs +1 ):
        print(f"Epoch {epoch_no} - Starting Training.")
        model=train(model, train_data_loader, loss_criterion, optimizer, device, epoch_no)
        print(f"Epoch {epoch_no} - Starting Testing.")
        test(model, test_data_loader, loss_criterion, device, epoch_no)
    # Save the trained model to the specified model directory.
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print("Completed Saving the Model")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
    parser.add_argument( "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)" )
    parser.add_argument( "--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)" )
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    
    main(args)