import time

import pandas as pd
from tqdm import tqdm


import multiprocessing as mp
from collections import Counter
from util import *
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

threshold = 0.9

def RemoveGit(str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str)


def clean_en_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)  # keep letter, digit and blank space
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


def get_shared_token_num_ratio(desc, deepseek_text):
    desc_cve_token = desc.split(' ')
    ds_text_token = deepseek_text.split(' ')
    c1 = Counter(desc_cve_token)
    c2 = Counter(ds_text_token)
    c3 = c1 & c2

    shared_num = len(c3.keys())
    shared_ratio = shared_num / (len(c1.keys()) + 1)
    c3_value = list(c3.values())
    if len(c3_value) == 0:
        c3_value = [0]
    return shared_num, shared_ratio, max(c3_value), sum(c3_value), np.mean(c3_value), np.var(c3_value)


def multi_vuln_commit_token(row):
    return get_shared_token_num_ratio(row['desc'], row['deepseek_text'])


def compute_similarity(args):
    group, cve = args

    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit(group['deepseek_text'])
    except Exception as e:
        group['deepseek_text'].fillna('', inplace=True)
        vectorizer.fit(group['deepseek_text'])

    similarity_scores = []

    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['desc']])
        ds_text_tfidf = vectorizer.transform([row['deepseek_text']])
        similarity = cosine_similarity(desc_tfidf, ds_text_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['commit'] = group['commit']
    similarity_data['commit_vuln_ds_tfidf'] = similarity_scores

    return similarity_data

def group_and_get_commit_feature(cve_name):
    df = pd.read_csv(f'../cache/{cve_name}_relevance_scores.csv')
    df['is_related'] = df['relevance_score'].apply(lambda score: 1 if score >= threshold else 0)


    all_connected_components = {}
    commit_in_group_rows = []
    for cve, group in df.groupby('cve'):
        related_group = group[group['is_related'] == 1]

        G = nx.Graph()                      
        commit1_list = group['commit1'].to_list()
        commit2_list = group['commit2'].to_list()
        all_commits = list(set(commit1_list + commit2_list))
        G.add_nodes_from(all_commits)       
        for _, row in group.iterrows():     
            if row['is_related'] == 1:
                G.add_edge(row['commit1'], row['commit2'])
        connected_components = list(nx.connected_components(G))  
        all_connected_components[cve] = connected_components

        component_mapping = {}  
        for i, component in enumerate(connected_components, start=1):
            for commit in component:
                component_mapping[commit] = i

    
        commit1_unique = group[['cve', 'commit1', 'label1']].drop_duplicates()
        commit2_unique = group[['cve', 'commit2', 'label2']].drop_duplicates()

        commit1_unique.columns = ['cve', 'commit', 'label']
        commit2_unique.columns = ['cve', 'commit', 'label']

        commit_in_group = pd.concat([commit1_unique, commit2_unique], ignore_index=True)
        commit_in_group = commit_in_group.drop_duplicates()
        commit_in_group['group_id'] = commit_in_group['commit'].apply(lambda x: component_mapping.get(x, 0))

        commit_in_group = commit_in_group.sort_values(by='group_id', ascending=True)  
        commit_in_group = pd.concat(
            [commit_in_group[commit_in_group['group_id'] != 0], commit_in_group[commit_in_group['group_id'] == 0]])
        commit_in_group = commit_in_group.reset_index(drop=True)

        commit_in_group_rows.append(commit_in_group)


    df_connected_components = pd.DataFrame(
        [(cve, str(connected_components)) for cve, connected_components in all_connected_components.items()],
        columns=['cve', 'connected_components']
    )

    df_group_id = pd.concat(commit_in_group_rows, ignore_index=True)

    # 1-1: load csv
    df_train = pd.read_csv(f'../cache/{cve_name}_top50.csv')
    df_train = df_train.merge(df_group_id[['cve', 'commit', 'group_id']], how='left', on=['cve', 'commit'])
    with mp.Pool(mp.cpu_count()) as pool:
        df_train['desc'] = list(tqdm(pool.imap(textProcess, df_train['desc']), total=len(df_train), desc='Processing desc'))


    df_deepseek = pd.read_csv(f'../cache/{cve_name}-deepseek.csv')
    df_deepseek = df_deepseek.dropna(subset=['is_patch'])
    df_deepseek = df_deepseek[df_deepseek['is_patch'] != 'YES/NO/UNKNOWN']
    mapping = {'YES': 1, 'UNKNOWN': 0.5, 'NO': 0}
    df_deepseek['patch_score'] = df_deepseek['is_patch'].map(mapping)
    df_deepseek['potential_addressed_vulnerability_types'].fillna('[]', inplace=True)
    df_deepseek['potential_addressed_vulnerability_types'] = df_deepseek['potential_addressed_vulnerability_types'].apply(eval)
    df_deepseek['potential_addressed_vulnerability_types_text'] = df_deepseek['potential_addressed_vulnerability_types'].apply(
        lambda x: f"It may address {', '.join(x)}." if x else "")
    df_deepseek['deepseek_text'] = df_deepseek['summarization'] + df_deepseek['potential_addressed_vulnerability_types_text']
    df_deepseek = df_deepseek[['cve', 'commit', 'deepseek_text', 'patch_score']]

    df_train = df_train.merge(df_deepseek, how='inner', on=['cve', 'commit'])
    with mp.Pool(mp.cpu_count()) as pool:
        df_train['deepseek_text'] = list(tqdm(pool.imap(textProcess, df_train['deepseek_text']), total=len(df_train),
                                            desc='Processing deepseek_text'))

    # 1-2: get feature
    print('Start train dataset processing at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    output_path = '../cache/'

    num_cores = 4

    df = df_train.copy()

    # generate features
    print('Start train dataset\'s features generating at',
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(multi_vuln_commit_token, [row for _, row in df.iterrows()]), total=len(df),
                            desc='vuln_ds-text shared words'))
    df['ds_shared_num'], df['ds_shared_ratio'], df['ds_max'], df['ds_sum'], df['ds_mean'], df['ds_var'] = zip(*results)

    single_cve_group = df.groupby('cve')
    with mp.Pool(num_cores) as pool:
        result_list = list(tqdm(pool.imap(compute_similarity, [(group, cve) for cve, group in single_cve_group]),
                                total=len(single_cve_group), desc='Computing deepseek text similarity scores'))
    tfidf_df = pd.concat(result_list, ignore_index=True)
    df = df.merge(tfidf_df, how='left', on=['cve', 'commit'])
    
    data_df = df.reset_index(drop=True)
    
    df_each = pd.read_csv(f'../cache/{cve_name}_commit_info.csv')
    merge_df = data_df.merge(df_each[['cve', 'commit']], how='inner', on=['cve', 'commit'])
    data_df = merge_df

    df_each = pd.read_csv(f'../cache/{cve_name}_feature.csv')
    merge_df = data_df.merge(df_each, how='inner', on=['cve', 'commit'])
    data_df = merge_df

    df_each = pd.read_csv(f'../cache/{cve_name}_commits.csv')
    merge_df = data_df.merge(df_each[['cve', 'commit', 'msg_text']], how='left', on=['cve', 'commit'])
    data_df = merge_df
    
    handcrafted_columns = ['cve', 'commit', 'group_id', 'desc', 'msg_text', 'deepseek_text',
                    'addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                    'cve_match', 'bug_match', 'issue_match', 'cwe_match',
                    'time_dis', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                    'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                    'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                    'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                    'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'commit_vuln_tfidf', 'commit_vuln_ds_tfidf',
                    'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                    'ds_shared_num', 'ds_shared_ratio', 'ds_max', 'ds_sum', 'ds_mean', 'ds_var', 'patch_score']    
    data_df = data_df[handcrafted_columns]
    print(data_df['group_id'].value_counts())
    data_df.to_csv('{}{}_feature-deepseek-text.csv'.format(output_path, cve_name), index=False)

if __name__ == '__main__':
    cve_name = "CVE-2018-6596"
    group_and_get_commit_feature(cve_name)