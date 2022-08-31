import os
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer, BertModel)
from models import *
from load_text import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from bert_process_THUCNews import feature_extraction
from cosformer import CosformerAttention 

# Training settings
parser = argparse.ArgumentParser()
#data
parser.add_argument('--limit_file_num', type=int, default=200000, help='limit the number of files per label.')
parser.add_argument('--limit_file_length', type=int, default=20, help='limit the number of lines per file.')
parser.add_argument('--feature_save_path', type=str, default='/home/data/toy/bert_feature_CEC/', help='path to save bert features.')
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

def code_labels():
    label_dict={}
    num = 0
    df = pd.read_csv('CEC_file_path.csv',index_col='path')
    for i in df.index:
        label = df.loc[i,'label']
        if label in label_dict.keys():
            pass
        else:
            label_dict[label] = num
            num+=1
    return label_dict

def save_feature(label_dict):
    df = pd.read_csv('CEC_file_path.csv',index_col='path')
    path = '/home/data/toy/raw_data/CEC-Corpus/raw corpus/allSourceText/'
    file_list = os.listdir(path)
    output_list = []
    label_list = []
    for file in tqdm(file_list):
        # print(file)
        file_name = file.strip('.txt')
        label = df.loc[file_name,'label']
        label_list.append(torch.tensor(label_dict[label]))
        text_list = []
        with open(path+file) as f:
            lines = f.readlines()
            for i in range(1,len(lines),2):
                text_list.append(lines[i].strip('\n'))
            f.close()
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
    torch.save(output_all,os.path.join(outpath,'feature.pl'))
    label_all = torch.stack(label_list,dim=0)
    torch.save(label_all,os.path.join(outpath,'label.pl'))

if __name__ == '__main__':
    # path = '/home/data/toy/raw_data/CEC-Corpus/CEC/'
    # df = pd.DataFrame()
    # folder_list = os.listdir(path)
    # for folder in folder_list:
    #     file_list = os.listdir(os.path.join(path,folder))
    #     for file in file_list:
    #         file_name = file.strip('.xml')
    #         df.loc[file_name,'label'] = folder
    #         df.loc[file_name,'xmlpath'] = os.path.join(path,folder,file)
    # df.to_csv('CEC_file_path.csv',index_label='path')
    with torch.no_grad():
        save_feature(code_labels())