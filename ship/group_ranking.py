import gc
import json
import os
import re
import logging
import time
from collections import OrderedDict

# 设置transformers缓存目录到有权限的位置
cache_dir = os.path.abspath('../transformers_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
print(f"Transformers cache directory set to: {cache_dir}")

import nltk.data
from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaConfig, RobertaTokenizer, AutoModel

# GPU设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 自动检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
MULTI_GPU = 1 if n_gpu > 1 else 0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
batch_size = 24

tqdm.pandas()


def get_rank(df1, sortby, ascending=False):
    gb = df1.groupby('cve')
    l = []
    for item1, item2 in gb:
        item2 = item2.reset_index()
        item2 = item2.sort_values(sortby + ['commit_list'], ascending=ascending)
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


def collate_fn(batch):
    input_ids_batch = []
    attention_mask_batch = []
    hc_feature_batch = []
    label_batch = []
    cve_batch = []
    commit_batch = []

    for input_ids_text_list, attention_mask_text_list, hc_feature, label, cve, commit in batch:
        input_ids_batch.append(input_ids_text_list)
        attention_mask_batch.append(attention_mask_text_list)
        hc_feature_batch.append(hc_feature)
        label_batch.append(label)
        cve_batch.append(cve)
        commit_batch.append(commit)

    return (input_ids_batch, attention_mask_batch, hc_feature_batch, label_batch, cve_batch, commit_batch)


class GroupRankingDataset(Dataset):
    def __init__(self, cve_name):
        df = pd.read_csv(f'../cache/{cve_name}_feature-deepseek-text.csv')
        
        df['mess'] = df['msg_text'] + df['deepseek_text']
        handcrafted_columns = ['issue_cnt', 'bug_cnt', 'cve_cnt',
                               'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                               'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                               'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                               'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                               'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                               'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf',
                               'commit_vuln_ds_tfidf',
                               'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                               'ds_shared_num', 'ds_shared_ratio', 'ds_max', 'ds_sum', 'ds_mean', 'ds_var',
                               'patch_score']  
        df[handcrafted_columns] = df[handcrafted_columns].to_numpy()
        df['hc_feature'] = df[handcrafted_columns].apply(lambda row: row.tolist(), axis=1)
        
        self.hc_dim = len(handcrafted_columns)

        # 按CVE分组处理数据
        grouped_data = []
        for cve, group in df.groupby('cve'):
            desc = group['desc'].tolist()[0]

            for group_id, sub_group in df.groupby('group_id'):
                commit_list = []
                mess_list = []
                hc_feature_list = []
                for _, row in sub_group.iterrows():
                    commit_list.append(row['commit'])
                    mess_list.append(row['mess'])
                    hc_feature_list.append(row['hc_feature'])

                p_data = {
                    'cve': cve,
                    'commit': commit_list,
                    'desc': desc,
                    'mess': mess_list,
                    'hc_feature': hc_feature_list,
                    'group_id': group_id
                }
                grouped_data.append(p_data)

        df = pd.DataFrame(grouped_data)

        self.cve = df['cve']
        self.commit = df['commit']
        self.desc = df['desc']
        self.mess = df['mess']
        self.handcrafted = df['hc_feature']
        self.group_id = df['group_id']

        model_path = '../pretrained_model/roberta-large'
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}, downloading from Hugging Face...")
            model_path = 'roberta-large'  # 使用 Hugging Face 模型名称

        self.text_tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

        gc.collect()

    def __len__(self):
        return len(self.group_id)

    def __getitem__(self, index):
        max_seq_length = 512

        desc = self.desc[index] if isinstance(self.desc[index], str) else ''
        input_ids_text_list = []
        attention_mask_text_list = []
        for each in self.mess[index]:
            mess = each if isinstance(each, str) else ''
            input_ids_text, attention_mask_text = convert_examples_to_features(desc, mess, self.text_tokenizer,
                                                                               max_seq_length)
            input_ids_text_list.append(input_ids_text)
            attention_mask_text_list.append(attention_mask_text)

        sample = (input_ids_text_list, attention_mask_text_list, self.handcrafted[index],
              torch.tensor(self.group_id[index]), self.cve[index], self.commit[index])

        return sample


class GroupRankingModel(nn.Module):
    def __init__(self, hc_dim=37, model_path='../pretrained_model/roberta-large'):
        super(GroupRankingModel, self).__init__()

        self.hc_dim = hc_dim
        self.s_dim = 32
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}, downloading from Hugging Face...")
            model_path = 'roberta-large'  # 使用 Hugging Face 模型名称
        
        config = RobertaConfig.from_pretrained(model_path, cache_dir=cache_dir)
        self.textEncoder = AutoModel.from_pretrained(model_path, config=config, cache_dir=cache_dir)

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.hc_dim, self.hc_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.s_dim + self.hc_dim, (self.s_dim + self.hc_dim) // 2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear((self.s_dim + self.hc_dim) // 2, (self.s_dim + self.hc_dim) // 4),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear((self.s_dim + self.hc_dim) // 4, (self.s_dim + self.hc_dim) // 8),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear((self.s_dim + self.hc_dim) // 8, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids_text, attention_mask_text, handcrafted, group_info, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[1] 
        text_output = self.fc1(text_output) 
        handcrafted = self.fc2(handcrafted)  
        combine_output = torch.cat([text_output, handcrafted], dim=-1)  
        hidden_vectors = self.mlp1(combine_output) 

        pooled_vectors = torch.zeros(len(group_info), hidden_vectors.size(-1)).to(device)

        for i, group in enumerate(group_info):
            group_vectors = hidden_vectors[group, :]  
            pooled_vectors[i, :] = torch.max(group_vectors, dim=0).values   

        logits = self.mlp2(pooled_vectors)

        prob = torch.softmax(logits, -1)
        if label is not None:
            loss = self.criterion(logits, label)
            return loss, prob
        else:
            return prob


def predict_group_ranking(cve_name, model_path=None, batch_size=16):
    if model_path is None:
        model_path = "../checkpoint_Phase3_model.bin"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    print(f"Loading dataset for {cve_name}...")

    try:
        dataset = GroupRankingDataset(cve_name)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None
    
    print(f"Loading trained model from: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, torch.nn.DataParallel):
            state_dict = checkpoint.module.state_dict()
        elif hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value

        if 'fc2.weight' in new_state_dict:
            trained_hc_dim = new_state_dict['fc2.weight'].shape[0]
            print(f"Model was trained with {trained_hc_dim} handcrafted features")
        else:
            print("Warning: Could not detect feature dimension from model, using dataset dimension")
            trained_hc_dim = dataset.hc_dim
            
    except Exception as e:
        print(f"Error analyzing model structure: {e}")
        trained_hc_dim = dataset.hc_dim

    model = GroupRankingModel(hc_dim=trained_hc_dim)
    
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise e
    
    # 将模型移动到设备，只有在使用CUDA且有多GPU时才使用DataParallel
    model = model.to(device)
    if MULTI_GPU and n_gpu > 1 and device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel with {n_gpu} GPUs")
    else:
        print(f"Using single device: {device}")
    
    prob_list = []
    group_id_list = []
    cve_list = []
    commit_list = []

    model.eval()
    print(f"Starting prediction for {len(dataset)} groups...")
    
    bar = tqdm(dataloader, total=len(dataloader))
    for step, batch in enumerate(bar):
        input_ids_text_list = batch[0]
        attention_mask_text_list = batch[1]
        handcrafted_list = batch[2]
        group_id_batch = batch[3]
        cve_batch = batch[4]
        commit_batch = batch[5]
        input_ids_text = []
        attention_mask_text = []
        handcrafted = []
        group_info = []
        group_id = []
        cve = []
        commit = []

        for i in range(len(input_ids_text_list)):
            group_info.append(list(range(len(input_ids_text), len(input_ids_text) + len(input_ids_text_list[i]))))
            input_ids_text.extend(input_ids_text_list[i])
            attention_mask_text.extend(attention_mask_text_list[i])
            handcrafted.extend(handcrafted_list[i])

            group_id.append(group_id_batch[i])
            cve.append(cve_batch[i])
            commit.append(commit_batch[i])

            if (i + 1 < len(input_ids_text_list)) and (len(input_ids_text) + len(input_ids_text_list[i+1]) <= 64):
                continue
            else:
                input_ids_text = torch.stack(input_ids_text)
                attention_mask_text = torch.stack(attention_mask_text)
                handcrafted = torch.tensor(handcrafted).float()

                input_ids_text = input_ids_text.to(device)
                attention_mask_text = attention_mask_text.to(device)
                handcrafted = handcrafted.to(device)

                with torch.no_grad():
                    prob = model(input_ids_text, attention_mask_text, handcrafted, group_info)

                    prob_list.append(prob.cpu().numpy())
                    group_id_list.append(group_id)
                    cve_list.append(cve)
                    commit_list.append(commit)

                input_ids_text = []
                attention_mask_text = []
                handcrafted = []
                group_info = []
                group_id = []
                cve = []
                commit = []

    torch.cuda.empty_cache()

    cve_list = np.concatenate(cve_list, 0)
    prob_list = np.concatenate(prob_list, 0)
    prob_list = prob_list[:, 1]
    group_id_list = np.concatenate(group_id_list, 0)
    # commit_list = np.concatenate(commit_list, 0)
    commit_list_flat = []
    for sublist in commit_list:
        if isinstance(sublist, list):
            commit_list_flat.extend([str(item) for item in sublist])
        else:
            commit_list_flat.append(str(sublist))

    result_data = {
        'cve': cve_list,
        'commit_list': commit_list_flat,
        'predict': prob_list,
        'group_id': group_id_list
    }
    
    result_df = pd.DataFrame(result_data)

    result_df['rank'] = get_rank(result_df, ['predict'], ascending=False)

    result_df['commit_list'] = result_df['commit_list'].apply(lambda x: x if isinstance(x, list) else [x])
    result_df = result_df.explode('commit_list', ignore_index=True)
    result_df.rename(columns={'commit_list': 'commit'}, inplace=True)

    output_file = f'../cache/{cve_name}_group_ranking_results.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # 获取排名第一的commit
    top_commit = result_df.loc[result_df['rank'] == 1]
    if not top_commit.empty:
        top_commit_info = top_commit.iloc[0]    
        return top_commit_info['commit']
    else:
        print("Warning: No top-ranked commit found!")
        return None


if __name__ == '__main__':
    # 示例用法
    cve_name = "CVE-2018-6596"
    
    print(f'Predicting group ranking for {cve_name} at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    prediction_result = predict_group_ranking(cve_name)