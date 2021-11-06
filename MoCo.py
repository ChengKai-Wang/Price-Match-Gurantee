"""
Use Kaggle Competition: Price Match Guarantee datasets
to train a Momentum Contrast network (CVPR 2019)
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
import random
import time
from torch import nn, device, Tensor
import torch
from torchvision.models import resnet18, resnext50_32x4d
import os
from PIL import  Image
from torchvision import transforms as T
from tqdm import tqdm
import configparser


def generate_pairs(groups):
    random.seed(time.time())
    datas = []
    for i in tqdm(groups, ncols= 80):
        for index, row in groups[i].iterrows():
            anchor = row.image
            ids = groups[i].image.tolist()
            if len(ids)>1:
                ids.remove(row.image)
                positive = random.choice(ids)
                datas.append((anchor, positive))
    return datas



class MoCo(nn.Module):
    def __init__(self, dim: int = 128, K: int = 1024, m: float = 0.99, T: float = 0.07)->None:
        super(MoCo, self).__init__()
        """ 
        dim: output embedding dimension 
        K: dictionary size 
        m: momentum parameter 
        T: temperature parameter 
        """
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = resnet18(pretrained=True)
        self.encoder_k =  resnet18(pretrained=True)
        # Replace the fc layer to make sure that the output dimension is as our expected
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        
        # Fixed the parameters of key encoder which is updated by momentum contrast
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not updated by gradient
            
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _momentum_update_key_encoder(self):
        # Momentum update to the key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    def _dequeue_and_enqueue(self, keys: Tensor):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Dequeue and enqueue. I have modified it for more general
        if ptr + batch_size > self.K:
            rest = self.K - ptr
            self.queue[:, ptr:] = keys[0: rest].T
            self.queue[:, 0: (batch_size-rest)] = keys[rest:].t()
            ptr = batch_size-rest
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.t()
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q: Tensor, im_k: Tensor):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # This part is partially reference from original MoCo released code

        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        q = nn.functional.normalize(q, dim=1)
        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)


        # Compute logits
        # Einstein sum is more intuitive
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # Apply temperature
        logits /= self.T

        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue
        if self.training:
            self._dequeue_and_enqueue(k)
            return logits, labels
        return logits, labels, q

class PairImages(Dataset):
    def __init__(self, train_pairs: list, path: str):
        self.pairs = train_pairs
        self.path = path
        self.transforms = T.Compose([
            T.Resize((256,256)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor()])
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        img1 = Image.open(os.path.join(self.path,img1))
        img2 = Image.open(os.path.join(self.path,img2))
        return self.transforms(img1), self.transforms(img2)

if __name__ == '__main__':
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    config_path = os.path.join(cur_dir, "config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)

    # Parameters settings
    # Data
    csv_path = config['data']['csv_path']
    img_path = config['data']['img_path']
    test_ratio = float(config['data']['ratio'])
    

    # Model
    batch_size = int(config['model']['batch_size'])
    embedding_dim = int(config['model']['dim'])
    epoch = int(config['model']['epoch'])
    device = device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Divide training data into train, validation and test
    df = pd.read_csv(csv_path)
    length = len(list(df.groupby('label_group')))
    train_group = dict(list(df.groupby('label_group'))[:int(test_ratio*length)+1])
    test_group = dict(list(df.groupby('label_group'))[int(test_ratio*length)+1:])

    train_pairs = generate_pairs(train_group)

    # Train: validation 8: 2
    valid_pairs = train_pairs[int(0.8*len(train_pairs)):]
    train_pairs = train_pairs[:int(0.8*len(train_pairs))]
    train_data = PairImages(train_pairs, img_path)
    valid_data = PairImages(valid_pairs, img_path)

    model = MoCo(embedding_dim).to(device)


    creterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
    valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle = False)
    print("------Start Training-----")
    for i in range(epoch):
        glob_loss = 0
        val_loss = 0
        model.train()
        for step, (anchor, positive) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            logits, labels = model(anchor.to(device, dtype=torch.float32), positive.to(device, dtype=torch.float32))
            if step >= 0:
                loss = creterion(logits, labels)
                loss.backward()
                optimizer.step()
                glob_loss += loss.data
        model.eval()
        with torch.no_grad():
            for step, (anchor, positive) in enumerate(valid_loader):
                logits, labels, q= model(anchor.to(device, dtype=torch.float32), positive.to(device, dtype=torch.float32))
                loss = creterion(logits, labels)
                val_loss += loss.data
        torch.save(model, './models/MoCo_'+str(i)+'.pt')
        print("epoch: %3d, train loss: %.4f"%(i+1, glob_loss))
        print("epoch: %3d, valid loss: %.4f"%(i+1, val_loss))
        with open("./log.txt", "a+") as logfile:
            logfile.write("epoch: %3d, train loss: %.4f\n"%(i+1, glob_loss))
            logfile.write("epoch: %3d, valid loss: %.4f\n"%(i+1, val_loss))
