import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from PIL import  Image
from torchvision import transforms as T
from torch import Tensor
from torchvision.models import resnet18
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn import manifold
from MoCo import MoCo
import configparser


        


def precision(y_pred, y_true):
    pos = 0
    for s in y_pred:
        if s in y_true:
            pos += 1
    return pos/len(y_pred)
def recall(y_pred, y_true, prec):
    pos = prec * len(y_pred)
    return pos/len(y_true)
def f1score(prec, rec):
    return 2*((prec*rec)/(prec+rec))

def score(pred, answers):
    recalls = 0
    precisions = 0
    f1scores = 0
    cnt = 0
    for posting_id in answers:
        y_pred = pred[posting_id]
        y_true = answers[posting_id]
        prec = precision(y_pred, y_true)
        precisions += prec
        rec = recall(y_pred, y_true, prec)
        recalls += rec
        f1scores += f1score(prec, rec)
        cnt += 1
    return precisions/cnt, recalls/cnt, f1scores/cnt

def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# Clustering with top k similar images and the similarity larger than threshold
def community_detection(embeddings,posting_id_dict,
                        threshold=0.75, 
                        batch = 100,
                        max_size=50):
    y_pred = {}
    y_pred_match_sim = {}
    cos_scores = pytorch_cos_sim(embeddings, embeddings)
    for i in range(len(embeddings)):
        top_val_large, top_idx_large = cos_scores[i].topk(k=max_size, largest=True)
        top_idx_large = top_idx_large[top_val_large >= threshold].tolist()
        top_val_large = top_val_large[top_val_large >= threshold].tolist()
        matches = []
        matches_sim = []
        for j, sentence_id in enumerate(top_idx_large):
            matches.append(posting_id_dict[sentence_id])
            matches_sim.append(top_val_large[j])
        y_pred[posting_id_dict[i]] = matches
        y_pred_match_sim[posting_id_dict[i]] = matches_sim
    return y_pred, y_pred_match_sim

if __name__ == '__main__':
    proDir = os.path.split(os.path.realpath(__file__))[0]
    print(proDir)
    config_path = os.path.join(proDir, "config.ini")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Test
    model_path = config['test']['model_path']
    """ Test Data Preprocessing """
    df = pd.read_csv(csv_path)
    length = len(list(df.groupby('label_group')))
    test_group = dict(list(df.groupby('label_group'))[int(0.6*length)+1:])
    
    posting_id_dict = {}
    id_file_dict = {} # id, filename reference table
    id_group_dict = {} # id, group reference table
    groups_elements_list = []



    cnt = 0
    for group in tqdm(test_group, ncols=80):
        groups_elements = []
        for index, row in test_group[group].iterrows():
            posting_id_dict[cnt] = row.posting_id
            groups_elements.append(row.posting_id)
            id_file_dict[row.posting_id] = row.image
            id_group_dict[row.posting_id] = row.label_group
            cnt += 1
        groups_elements_list.append(groups_elements)
    max_length = 0 # max size of group
    answers = {} # posting id and corresponding group list
    for i in tqdm(posting_id_dict, ncols=80):
        for j in groups_elements_list:
            if posting_id_dict[i] in j:
                if len(j)>max_length:
                    max_length = len(j)
                answers[posting_id_dict[i]] = j
    print(max_length)
    
    

    model = torch.load(model_path).to(device)
    model = model.eval()
    embedding_lists = []
    transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    with torch.no_grad():
        for it in tqdm(posting_id_dict, ncols=80):
            file_name = id_file_dict[posting_id_dict[it]]
            img = Image.open((os.path.join(img_path, file_name)))
            img = transforms(img).reshape(1, 3, 256, 256)
            out = model(img.to(device, dtype=torch.float32), img.to(device, dtype=torch.float32))[2][0]
            embedding_lists.append(out.to(torch.device("cpu")).tolist())

        print("embedding generated OK")
    
    """ Intra-inter Distance """
    print("--- Compute Intra-inter Distance ---")
    intra_distance = 0
    intra_nums = 0
    inter_distance = 0
    inter_nums = 0
    for i in tqdm(range(len(embedding_lists)), ncols=80):
        pos_id_i = posting_id_dict[i]
        group_i = id_group_dict[pos_id_i]
        embed_i = np.array(embedding_lists[i])
        for j in range(len(embedding_lists)):
            pos_id_j = posting_id_dict[j]
            group_j = id_group_dict[pos_id_j]
            embed_j = np.array(embedding_lists[j])
            if group_i == group_j:
                intra_nums += 1
                intra_distance += (1 - np.dot(embed_i, embed_j))
            else:
                inter_nums += 1
                inter_distance += (1 - np.dot(embed_i, embed_j))
    intra_distance = intra_distance/ intra_nums
    inter_distance = inter_distance/ inter_nums
    print("intra-class distance: %.4f"%(intra_distance))
    print("inter-class distance: %.4f"%(inter_distance))
    print("distance ratio: %.4f"%(inter_distance/intra_distance))
    
    """ test threshold """
    print("--- Test different thresholds ---")
    thres = []
    p = []
    r = []
    f = []
    for i in tqdm(range(500, 1000, 1), ncols=80):
        y_pred, y_pred_match_sim = community_detection(torch.tensor(embedding_lists), posting_id_dict, threshold=i/1000, max_size = max_length)
        prec, rec, fscore = score(y_pred, answers)
        thres.append(i/1000)
        p.append(prec)
        r.append(rec)
        f.append(fscore)

    max_fscore_index = f.index(max(f))
    print("Best thers: %.6f, precision: %.4f, recall: %.4f, f1score: %.4f"%(thres[max_fscore_index], p[max_fscore_index], r[max_fscore_index], f[max_fscore_index]))
    
    # plot the precision-recall curve
    plt.title('Result Analysis')
    plt.plot(thres, p, color='green', label='Precision')
    plt.plot(thres, r, color='red', label='Recall')
    plt.plot(thres, f,  color='blue', label='F1 score')
    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('score')
    plt.savefig('reuslts')
    
    