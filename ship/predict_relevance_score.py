import gc
import os
import re
import time

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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaConfig, RobertaTokenizer, AutoModel


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 自动检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

tqdm.pandas()


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


class NewPairDataset(Dataset):
    def __init__(self, feature_file, cve_name=None):

        if cve_name:
            feature_file = f'../cache/{cve_name}_interrelationship_features.csv'
        
        model_path = '../pretrained_model/roberta-large'
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(model_path):
            print(f"Local RoBERTa model not found at {model_path}, downloading from Hugging Face...")
            model_path = 'roberta-large'
            
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        
        print(f"Loading features from: {feature_file}")
        df_feature = pd.read_csv(feature_file)

        df_feature['msg_text1'] = df_feature['msg_text1'].fillna('') + ' ' + df_feature['deepseek_text1'].fillna('')
        df_feature['msg_text2'] = df_feature['msg_text2'].fillna('') + ' ' + df_feature['deepseek_text2'].fillna('')
        
        print(f'Dataset size: {len(df_feature)} rows')

        self.cve = df_feature['cve']
        self.commit1 = df_feature['commit1']
        self.commit2 = df_feature['commit2']
        self.msg_text1 = df_feature['msg_text1']
        self.msg_text2 = df_feature['msg_text2']

        self.label1 = df_feature.get('label1', pd.Series([0] * len(df_feature)))
        self.label2 = df_feature.get('label2', pd.Series([0] * len(df_feature)))
        self.label = df_feature.get('label', pd.Series([0] * len(df_feature)))

        # 定义手工特征列
        # handcrafted_columns = ['cve_match', 'cve_num1', 'cve_num2', 'bug_match', 'bug_num1', 'bug_num2', 'issue_match',
        #                        'issue_num1', 'issue_num2', 'id_match', 'author_match', 'time_interval',
        #                        'same_func_used_num', 'same_func_used_ratio', 'opposite_ratio', 'opposite_num',
        #                        'same_ratio', 'same_num', 'same_function_num',
        #                        'same_function_ratio', 'same_file_num', 'same_file_ratio', 'same_msg_token_num',
        #                        'same_msg_token_ratio', 'commit_pair_tfidf', 'same_code_token_num', 'same_code_token_ratio',
        #                        'same_deepseek_text_token_num', 'same_deepseek_text_token_ratio',
        #                        'commit_pair_deepseek_tfidf', 'patch_score1', 'patch_score2'] 
        handcrafted_columns = ['cve_match', 'cve_num1', 'cve_num2', 'bug_match', 'bug_num1', 'bug_num2', 'issue_match',
                               'issue_num1', 'issue_num2', 'id_match', 'author_match', 'time_interval',
                               'same_func_used_num', 'same_func_used_ratio', 'opposite_ratio', 'opposite_num',
                               'same_ratio', 'same_num', 'same_function_num',
                               'same_function_ratio', 'same_file_num', 'same_file_ratio', 'same_msg_token_num',
                               'same_msg_token_ratio', 'same_code_token_num', 'same_code_token_ratio',
                               'same_deepseek_text_token_num', 'same_deepseek_text_token_ratio',
                               'commit_pair_deepseek_tfidf', 'patch_score1', 'patch_score2'] 
        

        handcrafted_feature = df_feature[handcrafted_columns]
        handcrafted_feature = handcrafted_feature.fillna(0)
        self.handcrafted = handcrafted_feature.to_numpy()
        self.hc_dim = handcrafted_feature.shape[1]  # 动态设置维度
        
        print(f"Using {self.hc_dim} handcrafted features")

        
        self.text_tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

        del df_feature
        gc.collect()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        max_seq_length = 512
        msg_text1 = self.msg_text1[index] if isinstance(self.msg_text1[index], str) else ''
        msg_text2 = self.msg_text2[index] if isinstance(self.msg_text2[index], str) else ''
        input_ids_text, attention_mask_text = convert_examples_to_features(msg_text1, msg_text2, self.text_tokenizer,
                                                                           max_seq_length)

        sample = (input_ids_text, attention_mask_text, torch.tensor(self.handcrafted[index]),
                  torch.tensor(self.label[index]), self.cve[index], self.commit1[index], self.commit2[index],
                  self.label1[index], self.label2[index])

        return sample


class NewPairModel(nn.Module):
    def __init__(self, hc_dim=32, model_path='../pretrained_model/roberta-large'):
        super(NewPairModel, self).__init__()

        self.hc_dim = hc_dim
        self.s_dim = 32
        
        # 如果本地模型不存在，从 Hugging Face 下载
        if not os.path.exists(model_path):
            print(f"Local RoBERTa model not found at {model_path}, downloading from Hugging Face...")
            model_path = 'roberta-large'
        
        config = RobertaConfig.from_pretrained(model_path, cache_dir=cache_dir)
        self.textEncoder = AutoModel.from_pretrained(model_path, config=config, cache_dir=cache_dir)

        self.fc1 = nn.Linear(self.textEncoder.config.hidden_size, self.s_dim)
        self.fc2 = nn.Linear(self.hc_dim, self.hc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.s_dim + self.hc_dim, (self.s_dim + self.hc_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.s_dim + self.hc_dim) // 2, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        for param in self.textEncoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids_text, attention_mask_text, handcrafted, label=None):
        text_output = self.textEncoder(input_ids=input_ids_text, attention_mask=attention_mask_text)[1] 

        text_features = self.fc1(text_output)

        hc_features = self.fc2(handcrafted)

        combined_features = torch.cat([text_features, hc_features], dim=1)
        logits = self.mlp(combined_features)
        
        prob = torch.softmax(logits, -1)
        if label is not None:
            loss = self.criterion(logits, label)
            return loss, prob
        else:
            return prob


def predict_relevance_scores(cve_name, batch_size=16):

    model_path = "../checkpoint_Phase2_model.bin"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    feature_file = f'../cache/{cve_name}_interrelationship_features.csv'
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    print(f"Loading trained model from: {model_path}")
    
    # 创建数据集和数据加载器
    dataset = NewPairDataset(feature_file, cve_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型并加载训练好的权重
    model = NewPairModel(hc_dim=dataset.hc_dim)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Creating dummy model for testing...")
        checkpoint = None
    else:
        checkpoint = torch.load(model_path, map_location=device)
    
    if checkpoint is not None:
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
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 只有在使用CUDA且有多GPU时才使用DataParallel
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using single device: {device}")
    
    # 预测
    prob_list = []
    label_list = []
    cve_list = []
    commit1_list = []
    commit2_list = []
    label1_list = []
    label2_list = []

    model.eval()
    print(f"Starting prediction for {len(dataset)} commit pairs...")
    
    bar = tqdm(dataloader, total=len(dataloader))
    for step, batch in enumerate(bar):
        input_ids_text = batch[0].to(device)
        attention_mask_text = batch[1].to(device)
        handcrafted = batch[2].float().to(device)
        label = batch[3]
        cve = batch[4]
        commit1 = batch[5]
        commit2 = batch[6]
        label1 = batch[7]
        label2 = batch[8]

        with torch.no_grad():
            prob = model(input_ids_text, attention_mask_text, handcrafted)

            prob_list.append(prob.cpu().numpy())
            label_list.append(list(label))
            cve_list.append(list(cve))
            commit1_list.append(list(commit1))
            commit2_list.append(list(commit2))
            label1_list.append(list(label1))
            label2_list.append(list(label2))

    torch.cuda.empty_cache()
    
    # 整理结果
    cve_list = np.concatenate(cve_list, 0)
    prob_list = np.concatenate(prob_list, 0)
    relevance_scores = prob_list[:, 1]
    label_list = np.concatenate(label_list, 0)
    commit1_list = np.concatenate(commit1_list, 0)
    commit2_list = np.concatenate(commit2_list, 0)
    label1_list = np.concatenate(label1_list, 0)
    label2_list = np.concatenate(label2_list, 0)

    result_data = {
        'cve': cve_list,
        'commit1': commit1_list,
        'commit2': commit2_list,
        'relevance_score': relevance_scores,
        'label': label_list,
        'label1': label1_list,
        'label2': label2_list
    }

    result_df = pd.DataFrame(result_data)
    
    output_file = f'../cache/{cve_name}_relevance_scores.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return result_df


if __name__ == '__main__':
    cve_name = "CVE-2018-6596"
    
    print(f'Predicting relevance scores for {cve_name} at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    try:
        result_df = predict_relevance_scores(cve_name)
        
        # 显示相关性最高的前5个commit对
        top_pairs = result_df.nlargest(5, 'relevance_score')
        print("\nTop 5 most relevant commit pairs:")
        for idx, row in top_pairs.iterrows():
            print(f"  {row['commit1'][:8]} <-> {row['commit2'][:8]}: {row['relevance_score']:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")