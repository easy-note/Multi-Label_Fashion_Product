from cmath import nan
import os
import torch
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm

matplotlib.style.use('ggplot')

def clean_data(df):
    drop_indices, drop_images_id = [], []
    print('[INFO]: Checking if all images are present')
    for index, image_id in tqdm(df.iterrows()):
        if not os.path.exists(f"./data/images-all/{image_id.id}.jpg"):
            drop_indices.append(index)
            drop_images_id.append(f"{image_id.id}.jpg")
        # Nan 전처리
        if image_id.gender == '' \
            or image_id.articleType == '' \
            or image_id.season == '' \
            or image_id.usage == '':
            drop_indices.append(index)
            drop_images_id.append(f"{image_id.id}.jpg")
            
    print(f"[INFO]: Dropping indices: {drop_indices}")
    print(f"[INFO]: Dropping image ids: {drop_images_id}")
    df.drop(df.index[drop_indices], inplace=True)
    return df

# save the trained model to disk
def save_model(epochs, model, optimizer, criterion):
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './model.pth')

# save the train and validation loss plots to disk
def save_loss_plot(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss.jpg')
    plt.show()