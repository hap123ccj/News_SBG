# coding=utf-8
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer, BertModel)
from models import *
from load_text import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from cosformer import CosformerAttention 

# Training settings
parser = argparse.ArgumentParser()
#data
parser.add_argument('--limit_file_num', type=int, default=200000, help='limit the number of files per label.')
parser.add_argument('--limit_file_length', type=int, default=20, help='limit the number of lines per file.')
parser.add_argument('--feature_save_path', type=str, default='/home/data/toy/bert_feature/', help='path to save bert features.')
#model
parser.add_argument('--hidden_final', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--gpu', type=int, default=0, help='selecting the GPU [0,1] to allocate model.')

args = parser.parse_args()
nlines = args.limit_file_length
outpath = args.feature_save_path
if not os.path.exists(outpath):
    os.mkdir(outpath)
    os.mkdir(os.path.join(outpath,'separate'))

#使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# #使用CPU
# device="cpu"
# use_cuda =False

#load BERT
config = BertConfig.from_pretrained('./bert-base-chinese/config.json')
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/')
bertmodel = BertModel.from_pretrained('./bert-base-chinese/', config=config)
model = my_bert_model(args, bert_model=bertmodel)
torch.backends.cudnn.benchmark = True

if use_cuda:
    print("use cuda data")
    model.to(device)

def feature_extraction(model,inputs):
    model.eval()
    
    output = model(**inputs)
    return output

def save_label():
    path = 'THUCNews/'
    file_path_list = []
    label_list = []
    folder_list = os.listdir(path)
    for label,folder in enumerate(folder_list):
        file_list = os.listdir(os.path.join(path,folder))
        max_file_num = min(len(file_list),args.limit_file_num)
        for file in file_list[:max_file_num]:
            label_list.append(torch.tensor(label))
            file_path = os.path.join(path,folder,file)
            file_path_list.append(file_path)
    label_all = torch.stack(label_list,dim=0)
    torch.save(label_all,os.path.join(outpath,'label.pl')) # first version '/home/data/toy/bert_feature/label.pl'
    return file_path_list

def save_feature_separately(file_path_list):
    path_list = []
    for i in range(0,len(file_path_list),1000):
        end = min(i+1000,len(file_path_list))
        feature_path = os.path.join(outpath,'separate',f'feature_{i}-{end}.pl')
        if not os.path.exists(feature_path):
            file_path_list_seg = file_path_list[i:end]
            output_list =[] #extraced bert features 
            for file_path in tqdm(file_path_list_seg,desc=str(i)+'.../'+str(len(file_path_list))):
                file = open(file_path)
                text_list = file.readlines()
                file.close()
                dataset = preprocess(text_list,512,tokenizer)
                batch_size = min(len(text_list),nlines) #fisrt version is 10
                dataloader = DataLoader(dataset, batch_size = batch_size) # Trains with this batch size.
                for step, batch in enumerate(dataloader):
                    inputs = {'input_ids': batch[0].to(device),
                        'attention_mask': batch[1].to(device),
                        'token_type_ids': batch[2].to(device).unsqueeze(1)}
                    output = feature_extraction(model,inputs)
                    output.cpu()
                    torch.cuda.empty_cache()
                    output_list.append(output)
                    break
            output_all = pad_sequence(output_list,batch_first=True)
            # print(label_all)
            torch.save(output_all,feature_path)
        path_list.append(feature_path)
    return path_list





if __name__ == '__main__':
    with torch.no_grad():
        file_list = save_label()
        feature_path_list = save_feature_separately(file_list)
        feature_list =[]
        for file in tqdm(feature_path_list):
            feature_new = torch.load(file,map_location=torch.device('cpu'))
            feature_list.append(feature_new)
        features = torch.cat(feature_list,dim=0)
        torch.save(features,os.path.join(outpath,'feature.pl'))

    
    

        

