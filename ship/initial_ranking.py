import datetime
import os
import gc
import warnings
import json
import glob
import shutil
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
from util import *
import ast
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re
import logging
import time
from collections import OrderedDict
import nltk.data
from nltk import word_tokenize
from nltk.corpus import stopwords

# 设置transformers缓存目录到有权限的位置
cache_dir = os.path.abspath('../transformers_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
print(f"Transformers cache directory set to: {cache_dir}")

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel, AutoModel)
from transformers import AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup

# 检查CUDA是否可用，如果不可用则使用CPU
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

tqdm.pandas()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
stop_words.add('.')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def RemoveGit(str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str)

def clean_en_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)  
    text = ' '.join(text.split())
    return text

def textProcess(text):
    final = []
    text = RemoveGit(text)
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word not in stop_words]
        for word in word_tokens:
            final.append(str(stemmer.stem(word)))

    if len(final) == 0:
        text = ' '
    else:
        text = ' '.join(final)
    return text

def diffcodeProcess(diffcode):
    added_code = ''
    deleted_code = ''
    added_annotation = ''
    deleted_annotation = ''
    lines = diffcode.split('\n')
    for line in lines:
        if line.startswith('+') and not line.startswith('++'):
            line = line[1:].strip()
            added_code = added_code + line + ' '
        if line.startswith('-') and not line.startswith('--'):
            line = line[1:].strip()
            deleted_code = deleted_code + line + ' '

    return added_code, deleted_code, added_annotation, deleted_annotation

def convert_examples_to_features(desc, mess, tokenizer, max_seq_length):
    desc_token = tokenizer.tokenize(desc)
    mess_token = tokenizer.tokenize(mess)

    if len(desc_token) + len(mess_token) > max_seq_length - 3:
        if len(desc_token) > (max_seq_length - 3) / 2 and len(mess_token) > (max_seq_length - 3) / 2:
            desc_token = desc_token[:int((max_seq_length - 3) / 2)]
            mess_token = mess_token[:max_seq_length - 3 - len(desc_token)]
        elif len(desc_token) > (max_seq_length - 3) / 2:
            desc_token = desc_token[:max_seq_length - 3 - len(mess_token)]
        elif len(mess_token) > (max_seq_length - 3) / 2:
            mess_token = mess_token[:max_seq_length - 3 - len(desc_token)]
    combined_token = [tokenizer.cls_token] + desc_token + [tokenizer.sep_token] + mess_token + [tokenizer.sep_token]
    input_ids_text = tokenizer.convert_tokens_to_ids(combined_token)
    if len(input_ids_text) < max_seq_length:
        padding_length = max_seq_length - len(input_ids_text)
        input_ids_text += [tokenizer.pad_token_id] * padding_length
    input_ids_text = torch.tensor(input_ids_text)
    assert len(input_ids_text) == max_seq_length, 'Length of input_ids_text is error!'

    attention_mask_text = input_ids_text.ne(tokenizer.pad_token_id).to(torch.int64)
    return input_ids_text, attention_mask_text

def get_rank(df1, sortby, ascending=False):
    gb = df1.groupby('cve')
    l = []
    for item1, item2 in gb:
        item2 = item2.reset_index()
        item2 = item2.sort_values(sortby + ['commit'], ascending=ascending)
        item2 = item2.reset_index(drop=True).reset_index()
        l.append(item2[['index', 'level_0']])

    df1 = pd.concat(l)
    df1['rank'] = df1['level_0'] + 1
    df1 = df1.sort_values(['index'], ascending=True).reset_index(drop=True)
    return df1['rank']

def get_metrics_N(test, rankname='rank', N=10):
    cve_list = []
    cnt = 0
    total = []
    gb = test.groupby('cve')
    for item1, item2 in gb:
        item2 = item2.sort_values([rankname], ascending=True).reset_index(drop=True)
        idx = item2[item2.label == 1].index[-1] + 1
        if idx <= N:
            total.append(idx)
            cnt += 1
        else:
            total.append(N)
            cve_list.append(item1)
    return cnt / len(total), np.mean(total)

class SimpleDataset(Dataset):
    def __init__(self, cve, feature_file_path, cve_file_path):
        self.df = pd.read_csv(feature_file_path)
        
        df_cve = pd.read_csv(cve_file_path)
        df_cve = df_cve[['cve', 'desc']]
        self.df = self.df.merge(df_cve, how='left', on='cve')
        self.df = self.df.fillna('')
        
        print("Processing CVE descriptions...")
        with mp.Pool(mp.cpu_count()) as pool:
            self.df['desc'] = list(tqdm(pool.imap(textProcess, self.df['desc']), total=len(self.df),
                                                 desc='Processing cve_desc'))
        self.desc = self.df['desc']
        
        print("Processing commit messages...")
        with mp.Pool(mp.cpu_count()) as pool:
            self.df['msg_text'] = list(tqdm(pool.imap(textProcess, self.df['msg_text']), total=len(self.df),
                                                     desc='Processing msg_text'))
        self.mess = self.df['msg_text']
        
        print("Processing diff code...")
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(diffcodeProcess, self.df['diff_code']), total=len(self.df),
                                             desc='Processing diff_code'))
        self.df['added_code'], self.df['deleted_code'], self.df['added_an'], self.df['deleted_an'] = zip(*results)

        self.added_code = self.df['added_code']
        self.deleted_code = self.df['deleted_code']
        self.cve = self.df['cve']
        self.commit = self.df['commit']
        self.df['label'] = 0
        if 'label' not in self.df.columns:
            self.df['label'] = 0
        self.label = self.df['label']




        # 手工特征
        handcrafted_columns = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                        'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                        'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                        'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                        'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                        'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                        'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf',
                        'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                        'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']
        
        missing_columns = [col for col in handcrafted_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # 为缺失的列填充0
            for col in missing_columns:
                self.df[col] = 0
        
        handcrafted_feature = self.df[handcrafted_columns]
        self.handcrafted = handcrafted_feature.to_numpy() 

        # 加载tokenizer
        roberta_model_path = '../pretrained_model/roberta-large'
        codereviewer_model_path = '../pretrained_model/codereviewer'
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(roberta_model_path):
            print(f"Local RoBERTa model not found at {roberta_model_path}, downloading from Hugging Face...")
            roberta_model_path = 'roberta-large'
        
        if not os.path.exists(codereviewer_model_path):
            print(f"Local CodeReviewer model not found at {codereviewer_model_path}, downloading from Hugging Face...")
            codereviewer_model_path = 'microsoft/codereviewer'
        
        self.text_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path, cache_dir=cache_dir)
        self.code_tokenizer = AutoTokenizer.from_pretrained(codereviewer_model_path, cache_dir=cache_dir)

        gc.collect()

    def __len__(self):
        return len(self.cve)

    def __getitem__(self, index):
        max_seq_length = 512
        desc = self.desc[index] if isinstance(self.desc[index], str) else ''
        mess = self.mess[index] if isinstance(self.mess[index], str) else ''
        input_ids_text, attention_mask_text = convert_examples_to_features(desc, mess, self.text_tokenizer,
                                                                           max_seq_length)

        added_code = self.added_code[index] if isinstance(self.added_code[index], str) else ''
        deleted_code = self.deleted_code[index] if isinstance(self.deleted_code[index], str) else ''
        input_ids_diff, attention_mask_diff = convert_examples_to_features(added_code, deleted_code,
                                                                           self.code_tokenizer, max_seq_length)

        sample = (input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff,
                  torch.tensor(self.handcrafted[index]), torch.tensor(0),
                  self.cve[index], self.commit[index])

        return sample

class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()

        self.hc_dim = 39
        self.s_dim = 32
        
        # 设置模型路径
        roberta_model_path = '../pretrained_model/roberta-large'
        codereviewer_model_path = '../pretrained_model/codereviewer'
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(roberta_model_path):
            print(f"Local RoBERTa model not found at {roberta_model_path}, downloading from Hugging Face...")
            roberta_model_path = 'roberta-large'
        
        if not os.path.exists(codereviewer_model_path):
            print(f"Local CodeReviewer model not found at {codereviewer_model_path}, downloading from Hugging Face...")
            codereviewer_model_path = 'microsoft/codereviewer'
        
        config = RobertaConfig.from_pretrained(roberta_model_path, cache_dir=cache_dir)
        config.num_labels = 1
        self.textEncoder = AutoModel.from_pretrained(roberta_model_path, config=config, cache_dir=cache_dir)
        self.codeEncoder = AutoModelForSeq2SeqLM.from_pretrained(codereviewer_model_path, cache_dir=cache_dir).encoder

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.codeEncoder.config.hidden_size, self.s_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim * 2 + self.hc_dim, (self.s_dim * 2 + self.hc_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.s_dim * 2 + self.hc_dim) // 2, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True

    def forward(self, input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff, handcrafted, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[1] 
        code_output = self.codeEncoder(input_ids=input_ids_diff, attention_mask=attention_mask_diff).last_hidden_state[:, 0, :]

        text_output = self.fc1(text_output)   
        code_output = self.fc2(code_output)  
        combine_output = torch.cat([text_output, code_output], dim=-1)    
        combine_output = torch.cat([combine_output, handcrafted], dim=-1)  

        logits = self.mlp(combine_output)  

        prob = torch.softmax(logits, -1)
        if label is not None:
            loss = self.criterion(logits, label)
            return loss, prob
        else:
            return prob

def test_single_cve(model, test_dataloader, result_dir, cve_name):
    prob_list = []
    label_list = []
    cve_list = []
    commit_list = []

    model.eval()
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for step, batch in enumerate(bar):
        input_ids_text = batch[0].to(device)
        attention_mask_text = batch[1].to(device)
        input_ids_diff = batch[2].to(device)
        attention_mask_diff = batch[3].to(device)
        handcrafted = batch[4].float().to(device)
        label = batch[5].long().to(device)
        cve = batch[6]
        commit = batch[7]

        with torch.no_grad():
            prob = model(input_ids_text, attention_mask_text, input_ids_diff, attention_mask_diff, handcrafted)

            prob_list.append(prob.cpu().numpy())
            label_list.append(label.cpu().numpy())
            cve_list.append(list(cve))
            commit_list.append(list(commit))

    # torch.cuda.empty_cache()
    cve_list = np.concatenate(cve_list, 0)
    prob_list = np.concatenate(prob_list, 0)
    prob_list = prob_list[:, 1]
    label_list = np.concatenate(label_list, 0)
    commit_list = np.concatenate(commit_list, 0)

    p_data = {
        'cve': cve_list,
        'commit': commit_list,
        'predict': prob_list
    }

    result_csv = pd.DataFrame(p_data)
    # result_csv.to_csv(f'{result_dir}/rank_result_origin_{cve_name}.csv', index=False)
    result_csv['rank'] = get_rank(result_csv, ['predict'], ascending=False)
    result_csv.to_csv(f'{result_dir}/rank_result_{cve_name}.csv', index=False)

    print(f"Results for {cve_name}:")
    print(f"Total commits: {len(result_csv)}")
    print(f"Predictions saved to: {result_dir}/rank_result_{cve_name}.csv")
    
    # 显示前10个预测结果
    top_results = result_csv.sort_values('predict', ascending=False).head(10)
    print("\nTop 10 predictions:")
    print(top_results[['commit', 'predict', 'rank']].to_string(index=False))
    
    return result_csv

def initial_ranking(cve_name):
    model_path = "../checkpoint_Phase1_full_model.bin"
    feature_file_path = f"../cache/{cve_name}_feature.csv"
    cve_file_path = f"../cache/{cve_name}.csv"
    result_dir = "../cache"
    commit_file_path = f"../cache/{cve_name}_commits.csv"
    
    print(f'Start testing for CVE: {cve_name} at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    # 创建结果目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 1. 准备数据集
    print('1/3: Preparing dataset...')
    test_data = SimpleDataset(cve_name, feature_file_path, cve_file_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=32, num_workers=0)
    
    # 2. 加载模型
    print('2/3: Loading model...')
    model = NewModel()
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Creating dummy model for testing...")
        # 如果模型文件不存在，创建一个用于测试的模型
        checkpoint = None
    else:
        checkpoint = torch.load(model_path, map_location=device)
    
    if checkpoint is not None:
        # 处理DataParallel保存的模型
        if isinstance(checkpoint, torch.nn.DataParallel):
            state_dict = checkpoint.module.state_dict()
        elif isinstance(checkpoint, dict) and 'module.' in list(checkpoint.keys())[0]:
            # 移除'module.'前缀
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized model")
    else:
        print("Using randomly initialized model for testing")
    
    # 将模型移动到正确的设备
    model = model.to(device)
    
    # 只有在使用CUDA时才使用DataParallel
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using single device: {device}")
    
    # 3. 运行测试
    print('3/3: Running test...')
    result_csv = test_single_cve(model, test_dataloader, result_dir, cve_name)
    
    print('*************** Test completed ***************')
    top50 = result_csv.sort_values('predict', ascending=False).head(50)
    top50.drop(columns=['predict', 'rank', 'cve'], inplace=True)
    df1 = pd.read_csv(commit_file_path)
    top50 = top50.merge(df1, how='left', on='commit')
    df2 = pd.read_csv(cve_file_path)
    df2.drop(columns=['repo'], inplace=True)
    top50 = top50.merge(df2, how='left', on='cve')
    top50['author_username'] = top50['author']
    top50['committer_username'] = top50['committer']
    top50 = top50[['cve', 'repo', 'commit', 'msg_text', 'msg_url', 'diff_code', 'commit_time', 'desc', 'cwe', 'author_username', 'committer_username']]
    top50.to_csv(f'{result_dir}/{cve_name}_top50.csv', index=False)
    return result_csv

if __name__ == "__main__":
    # 配置参数
    cve_name = "CVE-2020-26249"
    
    # 运行测试
    result_csv = initial_ranking(cve_name)
