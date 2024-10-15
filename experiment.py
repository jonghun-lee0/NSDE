import os
import sys
import copy
import time
import random
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import uniform

from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

from torchsde import sdeint

import torchvision.models as models
from torchvision import datasets, transforms

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Setup argparse
parser = argparse.ArgumentParser(description='Training script for Neural ODE/SDE.')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, help='Seed for random number generators.', default=random.randint(0, 1000))
parser.add_argument('--dataname', type=str, help='Dataname for experiment.', default='mnist')
parser.add_argument('--noise_setting', type=str, help='Noise setting for model', default='none')
parser.add_argument('--stochastic_depth', type=int, help='Stochastic depth for model.', default=10)
parser.add_argument('--p', type=float, help='Dropout probability.', default=0)
parser.add_argument('--baseline', action='store_true', help='Whether to use the baseline model.')
parser.add_argument('--backbone', type=str, help='Baseline model type.', default='none')
parser.add_argument('--epochs', type=int, help='Training epochs.', default=100)
parser.add_argument('--lr', type=int, help='Default learning rate.', default=0.1)

args = parser.parse_args()

SEED = args.seed
seed_everything(SEED)
print("======")

## CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

## Setup
dataname = args.dataname

if not args.baseline:
    out_name = '_'.join([
        dataname, 
        str(args.noise_setting),
        str(args.stochastic_depth),
        str(args.p),
        str(SEED),
    ])
else:
    out_name = '_'.join([
        dataname, 
        str(args.backbone),
        str(args.p),
        str(SEED),
    ])

output_directory = 'output/{}/'.format(dataname)
output_path = os.path.join(output_directory, out_name)
print(output_path)
print("======")

## Check
if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if os.path.exists(output_path):
    print(f"Output file {output_path} already exists. Exiting...\n")
    sys.exit(1)

# 
class sde_model(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, input_channels, num_hidden_layers, noise_setting='none', time_embedding=True, p=0):
        super().__init__()

        # define params
        hidden_channels, output_channels = input_channels, input_channels

        if time_embedding:
            input_channels += 2
        self.time_embedding = time_embedding 

        # drift network
        self.linear_in = nn.Linear(input_channels, hidden_channels)            
        self.linears = nn.ModuleList(nn.Linear(hidden_channels, hidden_channels)
                                           for _ in range(num_hidden_layers))
        self.linear_out = nn.Linear(hidden_channels, output_channels)
        self.p = p

        # diffusion options
        self.noise_setting= noise_setting
        self.diffusion_in = nn.Linear(input_channels, hidden_channels)            
        if self.noise_setting != 'none':
            self.noise_t = nn.Sequential(nn.Linear(1 + int(time_embedding), hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, output_channels))
            
    def f(self, t, y):
        if torch.tensor(t).dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        if self.time_embedding:
            yy = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        else:
            yy = y
        z = self.linear_in(yy).relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).tanh()
        return z

    def g(self, t, y):
        if torch.tensor(t).dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        if self.time_embedding:
            tt = torch.cat((torch.sin(t), torch.cos(t)), dim=-1)
        else:
            tt = t
        if self.noise_setting == 'none':
            return torch.zeros_like(y).to(y.device)
        elif self.noise_setting == 'additive':
            return (self.noise_t(tt)).relu()
        elif self.noise_setting == 'multiplicative':
            return ((self.noise_t(tt)).relu() * y)
        elif self.noise_setting == 'dropout_TTN': # cvpr dropout diffusion with test time noise
            return np.sqrt(self.p / (1 - self.p)) * self.f(t, y)
        elif self.noise_setting == 'dropout': # cvpr dropout diffusion without test time noise
            if self.training: #train
                return np.sqrt(self.p / (1 - self.p)) * self.f(t, y)  
            else : #test
                return torch.zeros_like(y).to(y.device)


class SDE_Net(nn.Module):
    def __init__(self, layers=[4, 1], noise_setting='none', timesteps=torch.arange(0, 10), params = [], num_classes=1000):
        super(SDE_Net, self).__init__()

        # Input initial convolution
        self.inplanes = 64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Base convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        # Dynamically add additional convolutional layers as specified by layers[0]
        self.additional_convs = self._make_layer(self.inplanes, layers[0])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # SDE model
        self.feature_dim = self.inplanes*(2**layers[0])
        self.timesteps = timesteps
        self.p = params
        self.func = sde_model(input_channels=self.feature_dim, num_hidden_layers=layers[1], noise_setting=noise_setting, p=self.p)

        # Classifier
        self.fc = nn.Linear(self.feature_dim, num_classes)
    
    def _make_layer(self, planes, blocks):
        layers = []
        for _ in range(blocks):
            conv2d = nn.Conv2d(planes, planes * 2, kernel_size=3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm2d(planes * 2)
            layers.append(conv2d)
            layers.append(bn)
            layers.append(self.relu)
            planes *= 2  # Double the planes for the next additional layer
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Apply additional convolution layers
        for layer in self.additional_convs:
            x = layer(x)
            if isinstance(layer, nn.BatchNorm2d):
                x = self.maxpool(x)  # Apply max pooling after batch normalization

        # define initial value
        z0 = self.pool(x).view(x.size(0),-1)

        t = self.timesteps.float()
        dt = 0.1

        z_t = sdeint(sde=self.func,
                    y0=z0,
                    ts=t,
                    dt=dt,
                    method='euler')
            
        return self.fc(z_t[-1,:,:])


def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total_params, 'trainable': trainable_params}


def calculate_metrics(model, data_loader, device='cuda', topk=(1, 5)):
    model.eval()  # Set model to evaluation mode

    correct = {k: 0 for k in topk}
    total = 0
    total_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # Autocast during the forward pass
            with autocast():
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)

            total_loss += loss.item()
            
            # Calculate Top-k accuracies
            _, pred = output.topk(max(topk), 1, True, True)
            pred = pred.t()
            correct_pred = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct[k] += correct_pred[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()[0]
            total += target.size(0)

    total_loss /= len(data_loader)
    accuracies = {k: 100. * correct[k] / total for k in topk}
    return {
        'Loss': total_loss, 
        'Top-1_accuracy': accuracies[1],
        'Top-5_accuracy': accuracies[5] if 5 in accuracies else 'N/A'
    }


def data_loader(dataname, batch_size=32):
    # Common transformations
    # resize_transform = transforms.Resize((224, 224))
    
    norms = {
        'mnist': ((0.1307,), (0.3081,)),
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'stl10': ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        'tiny-imagenet': ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))  
    }
    
    sizes = {
        'mnist': 28,
        'cifar10': 32,
        'cifar100': 32,
        'stl10': 96,        
        'svhn': 32,
        'tiny-imagenet': 64  
    }

    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(sizes[dataname], scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),
    ])
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*norms[dataname]),
    ])

    if dataname == 'mnist':
        base_transform.transforms.insert(0, transforms.Grayscale(num_output_channels=3))

    kwargs = {'root': './data', 'download': True, 'transform': base_transform}
    if dataname == 'tiny-imagenet':
        root = os.path.join(kwargs['root'], 'tiny-imagenet-200')
        full_dataset = datasets.ImageFolder(os.path.join(root, 'train_preprocess'), transform=base_transform)
        test_dataset = datasets.ImageFolder(os.path.join(root, 'valid_preprocess'), transform=base_transform)
    elif dataname in ['svhn', 'stl10']:
        Dataset = getattr(datasets, dataname.upper())
        full_dataset = Dataset(split='train', **kwargs)
        test_dataset = Dataset(split='test', **kwargs)
    else:
        Dataset = getattr(datasets, dataname.upper())
        full_dataset = Dataset(train=True, **kwargs)
        test_dataset = Dataset(train=False, **kwargs)
    num_classes = 200 if dataname == 'tiny-imagenet' else 100 if dataname == 'cifar100' else 10

    # Stratified split
    try:
        targets = full_dataset.targets
    except:
        targets = full_dataset.labels    

    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=targets)
    train_dataset = Subset(full_dataset, train_idx)
    validation_dataset = Subset(full_dataset, val_idx)

    # Apply augmentation only to training dataset
    train_dataset.transform = transforms.Compose([augmentation, base_transform])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, validation_loader, test_loader, num_classes


# Set configuration
batch_sizes = {
    'mnist': 64,
    'cifar10': 128,
    'cifar100': 128,
    'stl10': 64,       
    'svhn': 128,
    'tiny-imagenet': 128,
}

train_loader, validation_loader, test_loader, num_classes = data_loader(dataname, batch_size=batch_sizes[dataname])

if not args.baseline:
    timesteps = torch.arange(0, args.stochastic_depth).to(device) # torch.linspace(0, 1, args.stochastic_depth).to(device)
    p = args.p
    
    model = SDE_Net(layers=[4, 1], noise_setting=args.noise_setting, timesteps=timesteps, params=p, num_classes=num_classes).to(device)

elif args.baseline:
    if args.backbone=='resnet18': 
        model = models.resnet18(weights=None, num_classes=num_classes).to(device)
        model.fc = nn.Sequential(nn.Dropout(args.p), nn.Linear(model.fc.in_features, num_classes)).to(device)
        
    elif args.backbone=='resnet34': 
        model = models.resnet34(weights=None, num_classes=num_classes).to(device)
        model.fc = nn.Sequential(nn.Dropout(args.p), nn.Linear(model.fc.in_features, num_classes)).to(device)

    elif args.backbone=='resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes).to(device)
        model.fc = nn.Sequential(nn.Dropout(args.p), nn.Linear(model.fc.in_features, num_classes)).to(device)
    
init_params(model)
print(f'Number of parameters compared to ResNet18: {count_parameters(model)} | {count_parameters(models.resnet18(weights=None, num_classes=num_classes))}\n')    
print(model)

scaler = GradScaler()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

# Train loop
result = []
best_validation_loss = float('inf')  
best_validation_accuracy = 0  
best_model = None  

epochs_since_improvement = 0
early_stopping_threshold = 10  # Stop if no improvement after 10 consecutive epochs

for epoch in range(1, args.epochs + 1):
    # Training step
    model.train()
    train_loss = 0
    clip_value = 1.0  # Define the maximum gradient norm

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Use autocast for the forward pass
        with autocast():
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Step the optimizer and update the scale for the next iteration
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        train_loss += current_loss

        # Calculate accuracy
        _, pred = torch.max(output, dim=1)
        current_accuracy = torch.tensor(torch.sum(pred == target).item() / len(pred))

        # Update tqdm bar with the latest loss and accuracy
        pbar.set_description(f'Epoch: {epoch} | Train Loss: {current_loss:.5f} | Accuracy: {current_accuracy*100:.2f}% ')
    train_loss /= len(train_loader)

    # Get validation metrics
    metrics = calculate_metrics(model, validation_loader, device=device)  
    print(f"Epoch: {epoch} | Valid Loss: {metrics['Loss']:.5f} | Top-1 Accuracy: {metrics['Top-1_accuracy']:.2f}% | Top-5 Accuracy: {metrics['Top-5_accuracy']:.2f}% ")
    validation_loss = metrics['Loss']
    validation_accuracy = metrics['Top-1_accuracy']
    
    # Update the learning rate scheduler
    scheduler.step(validation_loss)

    # Get test metrics
    metrics = calculate_metrics(model, test_loader, device=device)  
    print(f"Epoch: {epoch} | Test  Loss: {metrics['Loss']:.5f} | Top-1 Accuracy: {metrics['Top-1_accuracy']:.2f}% | Top-5 Accuracy: {metrics['Top-5_accuracy']:.2f}% ")
    test_loss = metrics['Loss']

    # Save result
    result.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'test_loss': test_loss,
        'metrics': metrics,
    })

    # Save the model if validation metric is improved
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_wts = copy.deepcopy(model.state_dict())  # Save the model state_dict
        epochs_since_improvement = 0  # reset counter
        print(f"-- New best model saved with validation loss: {best_validation_loss:.5f}")
    else:
        epochs_since_improvement += 1
        print(f"-- No improvement in validation metric for {epochs_since_improvement} epochs.")

    # Early stopping condition
    if (epoch >= 50) & (epochs_since_improvement >= early_stopping_threshold):
        print("Early stopping triggered.")
        break        

    # model.load_state_dict(best_model_wts)
    print("------")

# Print the best performance
validation_losses = np.array([r['validation_loss'] for r in result])
index_min_loss = np.argmin(validation_losses)
print(result[index_min_loss])

with open(output_path, 'wb') as f:
    pickle.dump([dataname, args, result], f)

print(f'Saved at {output_path}\n')