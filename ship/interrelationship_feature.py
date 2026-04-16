import datetime
import os
import gc
import warnings
import json
import glob
import shutil
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter
from util import *
import ast
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


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

with open('../data/ghsa_dict.pkl', 'rb') as f:
    ghsa_to_cve_dict = pickle.load(f)


def re_cve(item, msg_urls):
    cve_list = re.findall(r'(CVE-[0-9]{4}-[0-9]{1,7})', item, re.IGNORECASE)
    for url in msg_urls:
        ghsa_ids = re.findall(r'GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}', url)
        for ghsa_id in ghsa_ids:
            if ghsa_id in ghsa_to_cve_dict and ghsa_to_cve_dict[ghsa_id]:
                cve_list.extend(ghsa_to_cve_dict[ghsa_id])
            else:
                try:
                    r = requests.get(f'https://github.com/advisories/{ghsa_id}')
                    r.encoding = r.apparent_encoding
                    demo = r.text
                    soup = BeautifulSoup(demo, 'html.parser')
                    it = iter(soup.find_all('div', 'color-fg-muted'))
                    for tag in it:
                        if tag.string is not None:
                            text = tag.string.strip()
                            if re.match(r'CVE-[0-9]{4}-[0-9]{1,7}', text):
                                cve_list.append(text)
                except Exception as e:
                    with open('data/ghsa_error.txt', 'a') as f:
                        f.write(f'Error processing {ghsa_id}: {e}\n')
    return set(cve_list)


def get_url_id(url):
    return re.findall(r'\b\d+\b', url)


def re_bug(text, msg_urls):
    list1 = re.findall(r'\bbug[^a-zA-Z0-9,.]{0,3}([0-9]{1,7})[^0-9]', text, re.IGNORECASE)
    list2 = []
    for url in msg_urls:
        if 'bug' in url or 'Bug' in url or 'bid' in url:
            list2.extend(get_url_id(url))
    return set(list1 + list2)


def re_issue(item, msg_urls):
    list1 = re.findall('[iI]ssue[^0-9]{0,5}([0-9]{1,7})[^0-9]', item)
    list2 = []
    for url in msg_urls:
        found = re.findall(r".*/issues/.*?(\d+).*$", url)
        list2.extend(found)
    return set(list1 + list2)


def calculate_match(commit1, commit2, msg_text1, msg_text2, msg_url1, msg_url2, author1, committer1, author2, committer2):
    cve_match = 0
    issue_match = 0
    bug_match = 0
    id_match = 0
    author_committer_match = 0

    cve_set1 = re_cve(msg_text1, msg_url1)
    cve_set2 = re_cve(msg_text2, msg_url2)
    if cve_set1 & cve_set2:
        cve_match = 1

    bug_set1 = re_bug(msg_text1, msg_url1)
    bug_set2 = re_bug(msg_text2, msg_url2)
    if bug_set1 & bug_set2:
        bug_match = 1

    issue_set1 = re_issue(msg_text1, msg_url1)
    issue_set2 = re_issue(msg_text2, msg_url2)
    if issue_set1 & issue_set2:
        issue_match = 1

    for item in msg_url1:
        if commit2 in item:
            id_match = 1
    for item in msg_url2:
        if commit1 in item:
            id_match = 1
    if commit2[:7] in msg_text1 or commit1[:7] in msg_text2:
        id_match = 1

    if len({author1, committer1} & {author2, committer2}) > 0:
        author_committer_match = 1

    return cve_match, len(cve_set1), len(cve_set2), bug_match, len(bug_set1), len(bug_set2), \
        issue_match, len(issue_set1), len(issue_set2), id_match, author_committer_match


def feature_time(commit_time1, commit_time2):
    time_dis = abs((commit_time1 - commit_time2).days)
    return time_dis


def re_func(diff_code):
    diff_code_lines = diff_code.split('\n')
    func_list = []
    function_def_pattern = r'def\s+(\w+)\s*\('
    function_call_pattern = r'(\w+)\s*\('
    for line in diff_code_lines:
        if (line.startswith('+') and not line.startswith('++')) or (line.startswith('-') and not line.startswith('--')):
            if len(line[1:].strip()) < 2: 
                continue
            call_matches = re.findall(function_call_pattern, line)
            def_matches = re.findall(function_def_pattern, line)
            func_list += (call_matches + def_matches)

    return set(func_list)


def same_func_used(diff_code1, diff_code2):
    same_func_list = re_func(diff_code1) & re_func(diff_code2)
    min_len = min(len(re_func(diff_code1)), len(re_func(diff_code2)))

    return len(same_func_list), len(same_func_list) / min_len if min_len > 0 else 0


def same_modified_line(diff_code1, diff_code2):
    modified_lines_1 = 0
    modified_lines_2 = 0
    opposite_modified_ratio = 0.0
    same_modified_ratio = 0.0
    diff_code1_lines = diff_code1.split('\n')
    diff_code2_lines = diff_code2.split('\n')

    dict_code_plus = dict()
    dict_code_minus = dict()
    dict_list = [dict_code_plus, dict_code_minus]
    file = ''
    for line in diff_code1_lines:
        if line.startswith('diff --git'):
            filepath = line.split(' ')[-1].strip()[2:]
            file = filepath.split('/')[-1]

        if (line.startswith('+') and not line.startswith('++')) or (line.startswith('-') and not line.startswith('--')):
            if len(line[1:].strip()) < 2:  # diff code is too short
                continue
            modified_lines_1 += 1

            if line[0] == '+':
                p_or_m = 0

            else:
                p_or_m = 1

            len_line = len(line) - 1
            if len_line not in dict_list[p_or_m].keys():                    # no len_line key (1)
                dict_file_code = dict()
                dict_file_code[file] = 1
                dict_line = dict()
                dict_line[line[1:]] = dict_file_code
                dict_list[p_or_m][len_line] = dict_line
            elif line[1:] not in dict_list[p_or_m][len_line].keys():        # no line[1:] key (2)
                dict_file_code = dict()
                dict_file_code[file] = 1
                dict_list[p_or_m][len_line][line[1:]] = dict_file_code
            elif file not in dict_list[p_or_m][len_line][line[1:]].keys():  # no file key (3)
                dict_list[p_or_m][len_line][line[1:]][file] = 1
            else:                                                           # num += 1
                dict_list[p_or_m][len_line][line[1:]][file] += 1

    dict_same_num = dict()
    dict_opposite_num = dict()
    file = ''
    for line in diff_code2_lines:
        if line.startswith('diff --git'):
            filepath = line.split(' ')[-1].strip()[2:]
            file = filepath.split('/')[-1]

        if (line.startswith('+') and not line.startswith('++')) or (line.startswith('-') and not line.startswith('--')):
            if len(line[1:].strip()) < 2:  # diff code is too short
                continue
            modified_lines_2 += 1

            if line[0] == '+':
                same = 0
                opposite = 1
            else:
                same = 1
                opposite = 0

            len_line = len(line) - 1
            if len_line in dict_list[opposite].keys() and line[1:] in dict_list[opposite][len_line].keys() \
                    and file in dict_list[opposite][len_line][line[1:]].keys():  # opposite sign: +&- / -&+
                file_code = file + '\'s' + line[1:]
                if file_code not in dict_opposite_num.keys():
                    dict_opposite_num[file_code] = 1
                elif dict_opposite_num[file_code] < dict_list[opposite][len_line][line[1:]][file]:
                    dict_opposite_num[file_code] += 1
            elif len_line in dict_list[same].keys() and line[1:] in dict_list[same][len_line].keys() \
                    and file in dict_list[same][len_line][line[1:]].keys():     # same sign: +&+ / -&-
                file_code = file + '\'s' + line[1:]
                if file_code not in dict_same_num.keys():
                    dict_same_num[file_code] = 1
                elif dict_same_num[file_code] < dict_list[same][len_line][line[1:]][file]:
                    dict_same_num[file_code] += 1

    min_modified_lines = min(modified_lines_1, modified_lines_2)
    same_modified_ratio = sum(dict_same_num.values()) / min_modified_lines if min_modified_lines > 0 else 0
    opposite_modified_ratio = sum(dict_opposite_num.values()) / min_modified_lines if min_modified_lines > 0 else 0
    return opposite_modified_ratio, sum(dict_opposite_num.values()), same_modified_ratio, sum(dict_same_num.values()), \
        modified_lines_1, modified_lines_2


def convert_lowercase_remove_symbol(item):
    item = item.lower()
    item = re.sub(r'[^a-zA-Z0-9 ]', ' ', item)
    item = " ".join(item.split())
    return item


def tokenize_text(text):
    token_list = []
    text = convert_lowercase_remove_symbol(text)
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word not in stop_words]
        for word in word_tokens:
            token_list.append(str(stemmer.stem(word)))

    return token_list


def count_common_values(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    common_count = 0
    for value in counter1:
        if value in counter2:
            min_count = min(counter1[value], counter2[value])
            common_count += min_count

    return common_count


def get_num_ratio(msg_text1, diff_code1, msg_text2, diff_code2, deepseek_text1, deepseek_text2):
    msg_token1 = msg_text1.split(' ')
    deepseek_token1 = deepseek_text1.split(' ')
    diff_code_lines1 = diff_code1.split('\n')
    filepaths1, files1, funcs1, code_token1 = [], [], [], []

    msg_token2 = msg_text2.split(' ')
    deepseek_token2 = deepseek_text2.split(' ')
    diff_code_lines2 = diff_code2.split('\n')
    filepaths2, files2, funcs2, code_token2 = [], [], [], []

    for line in diff_code_lines1:
        if re.match(r'^diff\s+--git', line):
            filepath1 = line.split(' ')[-1].strip()[2:]
            filepaths1.append(filepath1)
            file1 = filepath1.split('/')[-1]
            files1.append(file1)
        elif line.startswith('@@ '):
            funcname1 = line.split('@@')[-1].strip()
            funcname1 = funcs_preprocess(funcname1)
            funcs1.append(funcname1)
        else:
            if line.startswith('+') and not line.startswith('++'):  
                code_token1.extend(tokenize_text(line))
            elif line.startswith('-') and not line.startswith('--'):  
                code_token1.extend(tokenize_text(line))

    for line in diff_code_lines2:
        if re.match(r'^diff\s+--git', line):
            filepath2 = line.split(' ')[-1].strip()[2:]
            filepaths2.append(filepath2)
            file2 = filepath2.split('/')[-1]
            files2.append(file2)
        elif line.startswith('@@ '):
            funcname2 = line.split('@@')[-1].strip()
            funcname2 = funcs_preprocess(funcname2)
            funcs2.append(funcname2)
        else:
            if line.startswith('+') and not line.startswith('++'): 
                code_token2.extend(tokenize_text(line))
            elif line.startswith('-') and not line.startswith('--'):  
                code_token2.extend(tokenize_text(line))

    same_function_num = 0
    for item in funcs1:
        if item in funcs2:
            same_function_num += 1
    min_len = min(len(funcs1), len(funcs2))
    same_function_ratio = same_function_num / min_len if min_len > 0 else 0

    same_file_num = 0
    for item in files1:
        if item in files2:
            same_file_num += 1
    min_len = min(len(files1), len(files2))
    same_file_ratio = same_file_num / min_len if min_len > 0 else 0

    same_msg_token_num = count_common_values(msg_token1, msg_token2)

    min_len = min(len(msg_token1), len(msg_token2))
    same_msg_token_ratio = same_msg_token_num / min_len if min_len > 0 else 0

    same_deepseek_token_num = count_common_values(deepseek_token1, deepseek_token2)


    min_len = min(len(deepseek_token1), len(deepseek_token2))
    same_deepseek_token_ratio = same_deepseek_token_num / min_len if min_len > 0 else 0

    same_code_token_num = count_common_values(code_token1, code_token2)



    min_len = min(len(code_token1), len(code_token2))
    same_code_token_ratio = same_code_token_num / min_len if min_len > 0 else 0

    return same_function_num, same_function_ratio, same_file_num, same_file_ratio, same_msg_token_num, \
        same_msg_token_ratio, same_code_token_num, same_code_token_ratio, same_deepseek_token_num, same_deepseek_token_ratio


def get_inter_commit_features(row):
    cve_match, cve_num1, cve_num2, bug_match, bug_num1, bug_num2, issue_match, issue_num1, issue_num2, id_match, \
        author_match = calculate_match(row['commit1'], row['commit2'], row['msg_text1'], row['msg_text2'],
                                       row['msg_url1'], row['msg_url2'], row['author1'], row['committer1'],
                                       row['author2'], row['committer2'])

    time_interval = feature_time(row['commit_time1'], row['commit_time2'])

    same_func_used_num, same_func_used_ratio = same_func_used(row['diff_code1'], row['diff_code2'])

    opposite_ratio, opposite_num, same_ratio, same_num, modified_lines_1, modified_lines_2 \
        = same_modified_line(row['diff_code1'], row['diff_code2'])

    same_function_num, same_function_ratio, same_file_num, same_file_ratio, same_msg_token_num, same_msg_token_ratio, \
        same_code_token_num, same_code_token_ratio, same_deepseek_token_num, same_deepseek_token_ratio = get_num_ratio(
        row['msg_text1'], row['diff_code1'], row['msg_text2'], row['diff_code2'], row['deepseek_text1'],
        row['deepseek_text2'])

    return cve_match, cve_num1, cve_num2, bug_match, bug_num1, bug_num2, issue_match, issue_num1, issue_num2, \
        id_match, author_match, time_interval, same_func_used_num, same_func_used_ratio, opposite_ratio, opposite_num, \
        same_ratio, same_num, modified_lines_1, modified_lines_2, same_function_num, same_function_ratio, \
        same_file_num, same_file_ratio, same_msg_token_num, same_msg_token_ratio, \
        same_code_token_num, same_code_token_ratio, same_deepseek_token_num, same_deepseek_token_ratio 


def compute_similarity(args):
    group, cve = args

    commit1_unique = group[['commit1', 'msg_text1']].drop_duplicates()
    commit2_unique = group[['commit2', 'msg_text2']].drop_duplicates()

    commit1_unique.columns = ['commit', 'text']
    commit2_unique.columns = ['commit', 'text']

    commits_texts = pd.concat([commit1_unique, commit2_unique], ignore_index=True)
    commits_texts = commits_texts.drop_duplicates()

    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit(commits_texts['text'])
    except Exception as e:
        commits_texts['text'].fillna('', inplace=True)
        vectorizer.fit(commits_texts['text'])

    similarity_scores = []

    for _, row in group.iterrows():
        commit1_tfidf = vectorizer.transform([row['msg_text1']])
        commit2_tfidf = vectorizer.transform([row['msg_text2']])
        similarity = cosine_similarity(commit1_tfidf, commit2_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['commit1'] = group['commit1']
    similarity_data['commit2'] = group['commit2']
    similarity_data['commit_pair_tfidf'] = similarity_scores

    return similarity_data


def compute_similarity_deepseek(args):
    group, cve = args

    commit1_unique = group[['commit1', 'deepseek_text1']].drop_duplicates()
    commit2_unique = group[['commit2', 'deepseek_text2']].drop_duplicates()

    commit1_unique.columns = ['commit', 'text']
    commit2_unique.columns = ['commit', 'text']

    commits_texts = pd.concat([commit1_unique, commit2_unique], ignore_index=True)
    commits_texts = commits_texts.drop_duplicates()

    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit(commits_texts['text'])
    except Exception as e:
        commits_texts['text'].fillna('', inplace=True)
        vectorizer.fit(commits_texts['text'])

    similarity_scores = []

    for _, row in group.iterrows():
        commit1_tfidf = vectorizer.transform([row['deepseek_text1']])
        commit2_tfidf = vectorizer.transform([row['deepseek_text2']])
        similarity = cosine_similarity(commit1_tfidf, commit2_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['commit1'] = group['commit1']
    similarity_data['commit2'] = group['commit2']
    similarity_data['commit_pair_deepseek_tfidf'] = similarity_scores

    return similarity_data


def get_interrelationship_feature(cve_name):

    df_train = pd.read_csv(f'../cache/{cve_name}_top50.csv')     
    df_deepseek = pd.read_csv(f"../cache/{cve_name}-deepseek.csv")
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
    print(f"After merging with DeepSeek data: {len(df_train)} rows")
    df_train['msg_text'] = df_train['msg_text'].fillna('')
    df_train['msg_url'] = df_train['msg_url'].apply(safe_literal_eval)
    df_train['msg_url'] = df_train['msg_url'].fillna('')
    df_train['diff_code'] = df_train['diff_code'].fillna('')
    df_train['commit_time'] = pd.to_datetime(df_train['commit_time'], format='%Y-%m-%d %H:%M:%S')
    df_train['deepseek_text'] = df_train['deepseek_text'].fillna('')

    with mp.Pool(mp.cpu_count()) as pool:
        df_train['msg_text'] = list(tqdm(pool.imap(textProcess, df_train['msg_text']), total=len(df_train),
                                            desc='Processing msg_text'))
    with mp.Pool(mp.cpu_count()) as pool:
        df_train['deepseek_text'] = list(tqdm(pool.imap(textProcess, df_train['deepseek_text']), total=len(df_train),
                                            desc='Processing deepseek_text'))

    commit_pair_df_train = pd.DataFrame()  
    for cve_id, group in df_train.groupby('cve'):
        commit_pair_rows = []
        
        # 获取所有commits，不区分正负样本
        all_commits = group.reset_index(drop=True)
        
        print(f"CVE {cve_id}: Processing {len(all_commits)} commits")

        # 生成所有可能的commit pairs
        for i in range(len(all_commits)):
            commit1 = all_commits.iloc[i]

            for j in range(i + 1, len(all_commits)):
                commit2 = all_commits.iloc[j]

                # 确保commit1的时间早于commit2
                # if commit1['commit_time'] > commit2['commit_time']:
                #     commit1, commit2 = commit2, commit1

                commit_pair_rows.append({
                    'cve': commit1['cve'],
                    'commit1': commit1['commit'],
                    'commit2': commit2['commit'],
                    'label1': 0,  # 使用get方法避免KeyError
                    'label2': 0,
                    'label': 0,  # 默认标签，可以根据需要修改
                    'msg_text1': commit1['msg_text'],
                    'msg_text2': commit2['msg_text'],
                    'msg_url1': commit1['msg_url'],
                    'msg_url2': commit2['msg_url'],
                    'diff_code1': commit1['diff_code'],
                    'diff_code2': commit2['diff_code'],
                    'commit_time1': commit1['commit_time'],
                    'commit_time2': commit2['commit_time'],
                    'author1': commit1['author_username'],
                    'author2': commit2['author_username'],
                    'committer1': commit1['committer_username'],
                    'committer2': commit2['committer_username'],
                    'deepseek_text1': commit1['deepseek_text'],
                    'deepseek_text2': commit2['deepseek_text'],
                    'patch_score1': commit1['patch_score'],
                    'patch_score2': commit2['patch_score']
                })

        commit_pair_df_tmp = pd.DataFrame(commit_pair_rows)
        commit_pair_df_train = pd.concat([commit_pair_df_train, commit_pair_df_tmp])


    # 添加数据检查
    if len(commit_pair_df_train) == 0:
        print("Warning: No commit pairs generated. Please check your data.")
        return pd.DataFrame()
    
    print(f"Total commit pairs generated: {len(commit_pair_df_train)}")

    print('Start train dataset processing at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    output_path = '../cache'   
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = commit_pair_df_train
    num_cores = 4

    print('Start features generating at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # 计算基础特征
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(get_inter_commit_features, [row for _, row in df.iterrows()]), 
                           total=len(df), desc='commit pair feature'))
    
    df['cve_match'], df['cve_num1'], df['cve_num2'], df['bug_match'], df['bug_num1'], df['bug_num2'], \
        df['issue_match'], df['issue_num1'], df['issue_num2'], df['id_match'], df['author_match'], \
        df['time_interval'], df['same_func_used_num'], df['same_func_used_ratio'], \
        df['opposite_ratio'], df['opposite_num'], df['same_ratio'], df['same_num'], \
        df['modified_lines_1'], df['modified_lines_2'], df['same_function_num'], df['same_function_ratio'], \
        df['same_file_num'], df['same_file_ratio'], df['same_msg_token_num'], df['same_msg_token_ratio'], \
        df['same_code_token_num'], df['same_code_token_ratio'], df['same_deepseek_text_token_num'], \
        df['same_deepseek_text_token_ratio'] = zip(*results)
    # 计算TF-IDF相似度
    single_cve_group = df.groupby('cve')
    with mp.Pool(num_cores) as pool:
        result_list = list(tqdm(pool.imap(compute_similarity, [(group, cve) for cve, group in single_cve_group]),
                               total=len(single_cve_group), desc='Computing similarity scores'))
    tfidf_df = pd.concat(result_list, ignore_index=True)

    df = df.merge(tfidf_df, how='left', on=['cve', 'commit1', 'commit2'])
    # 计算DeepSeek TF-IDF相似度
    with mp.Pool(num_cores) as pool:
        result_list = list(tqdm(pool.imap(compute_similarity_deepseek, [(group, cve) for cve, group in single_cve_group]),
                               total=len(single_cve_group), desc='Computing deepseek similarity scores'))
    tfidf_df = pd.concat(result_list, ignore_index=True)
    df = df.merge(tfidf_df, how='left', on=['cve', 'commit1', 'commit2'])

    # 直接保存最终结果
    output_filename = f'{cve_name}_interrelationship_features.csv'
    df.drop(columns=['msg_url1', 'msg_url2', 'diff_code1', 'diff_code2', 'commit_time1', 'commit_time2', 'author1', 'author2', 'committer1', 'committer2'], inplace=True)
    
    df.to_csv(os.path.join(output_path, output_filename), index=False)
    
    print(f'Features saved to {os.path.join(output_path, output_filename)}')
    
    return df

if __name__ == "__main__":
    cve_name = "CVE-2018-6596"  # Example CVE name
    result_df = get_interrelationship_feature(cve_name)
    print("Feature extraction completed!")