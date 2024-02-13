import pandas as pd
import torch
import torch.optim as optim

from dataset import train_val_split, FashionDataset
from torch.utils.data import DataLoader
from models import MultiHeadResNet50
from tqdm import tqdm
from loss_functions import loss_fn
from utils import save_model, save_loss_plot

from collections import defaultdict
from evaluation import label_based_eval

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiHeadResNet50(pretrained=True, requires_grad=False).to(device)

# learning parameters
lr = 0.001
optimizer = optim.Adam(params=model.parameters(), lr=lr)
criterion = loss_fn
batch_size = 32
epochs = 1

df = pd.read_csv('./data/styles.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], keep_default_na=False)
train_data, val_data, test_data = train_val_split(df)
print(f"[INFO]: Number of training sampels: {len(train_data)}")
print(f"[INFO]: Number of validation sampels: {len(val_data)}")
print(f"[INFO]: Number of test sampels: {len(test_data)}")

# training and validation dataset
train_dataset = FashionDataset(train_data, is_train=True)
val_dataset = FashionDataset(val_data, is_train=False)
test_dataset = FashionDataset(test_data, is_train=False)

# training and validation data loader
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# training function
def train(model, dataloader, optimizer, loss_fn, dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        articleType = data['articleType'].to(device)
        season = data['season'].to(device)
        usage = data['usage'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (gender, articleType, season, usage)
        loss = loss_fn(outputs, targets)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss

# validation function
def validate(model, dataloader, loss_fn, dataset, device):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        articleType = data['articleType'].to(device)
        season = data['season'].to(device)
        usage = data['usage'].to(device)
        
        outputs = model(image)
        targets = (gender, articleType, season, usage)
        loss = loss_fn(outputs, targets)
        val_running_loss += loss.item()
        
    val_loss = val_running_loss / counter
    return val_loss

# test function
def test(model, dataloader, dataset, device):
    model.eval()
    counter = 0
    
    outputs_dict = defaultdict(list)
    targets_dict = defaultdict(list)
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        articleType = data['articleType'].to(device)
        season = data['season'].to(device)
        usage = data['usage'].to(device)
        
        outputs = model(image)
                
        outputs_dict['gender'].extend(torch.argmax(outputs[0], dim=1))
        outputs_dict['articleType'].extend(torch.argmax(outputs[1], dim=1))
        outputs_dict['season'].extend(torch.argmax(outputs[2], dim=1))
        outputs_dict['usage'].extend(torch.argmax(outputs[3], dim=1))
        
        targets_dict['gender'].extend(gender)
        targets_dict['articleType'].extend(articleType)
        targets_dict['season'].extend(season)
        targets_dict['usage'].extend(usage)
    
    label_based_eval(outputs_dict, targets_dict)

# start the training
train_loss, val_loss = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_dataloader, optimizer, loss_fn, train_dataset, device
    )
    val_epoch_loss = validate(
        model, val_dataloader, loss_fn, val_dataset, device
    )
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    
test(model, test_dataloader, test_dataset, device)

# save the model to disk
save_model(epochs, model, optimizer, criterion)
# save the training and validation loss plot to disk
save_loss_plot(train_loss, val_loss)

