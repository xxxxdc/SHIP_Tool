import re
import pickle
import time
import enum
import string
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


def savefile(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    
def readfile(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data



def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def recog_commit(line, index):
    ret = re.search(r'commit ([\w+]*)', line)
    if ret:
        return ret.group(1)[:7]
    else:
        print('commit', index, line)
        return None
  
  
def recog_author(line):
    ret = re.search(r'Author: .* <(.*)>', line)
    if ret:
        return ret.group(1)
    else:
        print(line)
        return None
    return None
  

def recog_time(line):
    line = line[8:-7]
    try:
        struct_time = time.strptime(line,"%a %b %d %H:%M:%S %Y")
    except:
        return 0
    return '{:04}{:02}{:02}'.format(struct_time.tm_year, struct_time.tm_mon, struct_time.tm_mday)  


def recog_mess(lines, index):
    ret = []
    lens = len(lines)
    while index < lens and \
               not lines[index].startswith('commit') and \
               not lines[index].startswith('diff --git') and \
               not lines[index].startswith('@@ '):
        line = lines[index].strip()
        if len(line) >= 1:                                               # commit 信息
            ret.append(lines[index])
        index += 1
    return ' '.join(ret), index


def recog_filepath(line):
    filepath = line.split(' ')[-1].strip()[2:]                   # 文件路径
    return filepath


def recog_hunk(line):
    funcname = line.split('@@')[-1].strip()   # hunk header-函数名
    return funcname


def recog_code(lines, index):
    codes, addcodes, delcodes = '', '', ''
    addline, delline = 0, 0
    lens = len(lines)
    while index < lens and \
               not lines[index].startswith('commit') and \
               not lines[index].startswith('diff --git') and \
               not lines[index].startswith('@@ '):
        line = lines[index].strip()
        if len(line) >= 1:
            codes+= line+' '                   # 上下文代码行
            if line.startswith('+'):
                addcodes+= line[2: ]+' '      # 增加代码行
                addline += 1
            elif line.startswith('-'):
                delcodes+= line[2: ]+' '       # 删除代码行
                delline += 1
        index += 1
    return codes, addcodes, delcodes, addline, delline, index


def get_repo_total_data(lines, reponame):
    """
    处理文本内容，返回数据
    reponame —— 仓库名称
    commit —— commit sha 前7位
    author —— 邮箱表示 —— Delete
    date —— YearMonthDay共8位字符串表示
    mess —— ' '.join(list(信息))
    filepaths —— ' '.join(list(文件路径))
    funcs —— ' '.join(list(函数名))
    codes —— 总代码(含diff上下5行代码)，list(总代码)
    addcodes —— 增加的代码，list(增加代码)
    delcodes —— 减少的代码，list(减少代码)
    addlines —— 总增加代码行数
    dellines —— 总减少代码行数
    """
    index = 0
    lens = len(lines)
    total_data = []
    while index < lens:
        temp = []
        commit = recog_commit(lines[index].strip(), index)
        index += 1
        if lines[index].strip().startswith('Merge'):
            index += 1
        # author = recog_author(lines[index].strip())
        recog_author(lines[index].strip())
        index += 1
        date = recog_time(lines[index])
        index += 1
        mess, index = recog_mess(lines, index)  # mess 是列表
        # files, filepaths, funcs = [], [], []
        filepaths, funcs = [], []
        codes, addcodes, delcodes = [], [], []
        addlines, dellines = 0, 0        
        while index < lens and lines[index].startswith('diff --git'):
            filepath = recog_filepath(lines[index])
            # file, filepath = recog_file(lines[index])
            # files.append(file)
            filepaths.append(filepath)
            index += 1
            while not  lines[index].startswith('index') and \
                     not lines[index].startswith('commit') and \
                     not lines[index].startswith('diff --git') and \
                     not lines[index].startswith('@@ '):
                index += 1
            if lines[index].startswith('index'): index += 1
            if lines[index].startswith('Binary files '): index += 1
            if lines[index].startswith('--- '): index += 1
            if lines[index].startswith('+++ '): index += 1
            if len(lines[index].strip()) == 0: index += 1
            # 不能直接+4，因为有的会多一行"new file mode xxxx" /delete file mode /rename 等等
            # print(index, lines[index])
            while index < lens and lines[index].startswith('@@ -'):
                funcname = recog_hunk(lines[index])
                funcs.append(funcname)
                index += 1
                code, addcode, delcode, addline, delline, index = recog_code(lines, index)
                codes.append(code)
                addcodes.append(addcode)
                delcodes.append(delcode)
                addlines += addline
                dellines  += delline
                # print(index, lines[index])
        # temp = [reponame, commit, author, date, mess, filepaths, funcs, codes, addcodes, delcodes, addlines, dellines]
        temp = [reponame, commit, date, mess, ' '.join(filepaths), ' '.join(funcs), ' '.join(codes), ' '.join(addcodes), ' '.join(delcodes), addlines, dellines]
        total_data.append(temp)
    return total_data


def funcs_preprocess(item):
    keyword = ['auto', 'double', 'int', 'struct', 'break', 'else', 'long', 'switch', 
          'case', 'enum', 'register', 'typedef', 'char', 'extern', 'return', 'union', 
          'const', 'float', 'short', 'unsigned', 'continue', 'for', 'signed', 'void',
          'default', 'goto', 'sizeof', 'volatile', 'do', 'if', 'while', 'static']
    # ret = re.split(r'[^0-9a-zA-Z]', item)     # annotated by me
    ret = re.split(r'\(', item)[0]              # added by me
    ret = re.findall(r'\b[a-zA-Z0-9_]+\b', ret) # added by me
    ret = list(set(ret))
    ret = [item for item in ret if item and item not in keyword]
    return ' '.join(ret)


def string_preprocess(ret):
    ret = ret.replace(r"\r\n", ' ').replace(r"\n", ' ').replace(r"\r", ' ')
    ret = re.sub(r' +', ' ', ret) 
    return ret

#  --------------------  特征处理函数  --------------------
def As_in_B(As: list, B:str):
    """
    获取A列表中元素item有多少出现在字符串B中
    """
    cnt = 0
    for A in As:
        if A in B:
            cnt += 1
    return cnt


def re_search(query: str, item: str):
    """
    正则匹配，有返回List，没有返回None
    """
    return re.findall(query, item)
    

def sns_countplot(x_data, hue_data, other_data, data):
    """
    作图函数，仍有些问题
    """
    g = sns.countplot(x = data[x_data], hue=data[hue_data])
    grouped_values = data.groupby([x_data, hue_data])[other_data].count().reset_index()
    for index,row in grouped_values.iterrows():
        g.text(row[x_data] + 0.2 - 0.4 * (index%2==0), row.cve, row.cve ,color="black",ha="center")
        

def get_files(x):
    return list(set([item.split('/')[-1].strip()  for item in x]))


def max_union(array1, array2):
    array1 = set(array1)
    array2 = set(array2)
    return len(array1 & array2) / max(len(array1), len(array2))


def union_token(*array_list):
    arr = set()
    for array in array_list:
        arr = arr | set(array)
    return arr


def inter_token(*array_list):
    arr = array_list[0]
    for array in array_list:
        arr = arr & array
    return arr


def union_list(*items):
    ret = []
    for item in items:
        ret.extend(item)
    return ret

#  -----------------------  Token 分词  -----------------------
class StateType(enum.IntEnum):
    INITIAL_STATE = 0
    UPPERCASE_STATE = 1
    LOWERCASE_STATE = 2
    NUMBER_STATE = 3
    SPECIAL_STATE = 4
    
    
def line_to_tokens(code):
    """
    采用简单的字符类型的方式将代码进行切分
    upper | upper lower
    upper | number
    upper | special
    lower | upper
    lower | number
    lower | special
    number | upper
    number | lower
    number | special
    special | upper
    special | lower
    special | number
    结果示例："foo  ,1" -> ["foo", "  ", ",", "1"]
    """
    # normal state transitions that will result in splitting
    normal_transitions = [
      (StateType.UPPERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.UPPERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.LOWERCASE_STATE, StateType.UPPERCASE_STATE),
      (StateType.LOWERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.LOWERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.NUMBER_STATE, StateType.UPPERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.LOWERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.SPECIAL_STATE),
      (StateType.SPECIAL_STATE, StateType.UPPERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.LOWERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.NUMBER_STATE)]
    # output, state
    tokens = []
    state = StateType.INITIAL_STATE
    next_state = None
    memory = []
    for i, inputchar in enumerate(code):
        if inputchar.isupper():
            next_state = StateType.UPPERCASE_STATE
        elif inputchar.islower():
            next_state = StateType.LOWERCASE_STATE
        elif inputchar.isdigit():
            next_state = StateType.NUMBER_STATE
        else:
            next_state = StateType.SPECIAL_STATE

    # splitting cases
        if (state, next_state) in normal_transitions:
            tokens.append(''.join(memory))   # TheShape  -  The存储起来 Shape
            memory = []
        elif (state, next_state) == (StateType.UPPERCASE_STATE,
                                 StateType.LOWERCASE_STATE) and len(memory) > 1: # VSShape - VS Shape
            tokens.append(''.join(memory[:-1]))  
            memory = [memory[-1]]
        elif (state, next_state) == (StateType.SPECIAL_STATE,
                                 StateType.SPECIAL_STATE):
            if inputchar in [' ', '\t'] and inputchar == code[i-1]:   # 如果是空格或者\t 并且前一个字符也是，那么
                if len(memory) >= 20:   # 如果长度大于20，直接生成一个token
                    tokens.append(''.join(memory))
                    memory = []
            elif inputchar.isspace() or code[i-1].isspace(): # 如果是空格直接停止
                tokens.append(''.join(memory))
                memory = []

    # put inputchar into memory, always
        memory.append(inputchar)
        state = next_state
    if memory:
        tokens.append(''.join(memory))
    return tokens


def to_token(line, useful_token=None, unuseful_token=None):
    final_token = []   # 最后的token序列
    lmtzr = WordNetLemmatizer()
    stopwords_en = stopwords.words('english')
    
    tokens = re.split('[^0-9a-zA-Z]+', line)
    ret = []
    for token in tokens:
        ret.extend(line_to_tokens(token))
    # tokens = line_to_tokens(line)
    if unuseful_token:
        for token in ret:
            token_lower = token.lower()
            if token_lower in unuseful_token:
                ret.remove(token)

    if useful_token:
        for token in ret:
            token = token.lower()
            if token not in useful_token:
                continue
            token = lmtzr.lemmatize(token, 'v')   
            final_token.append(token) 
    else:
        for token in ret:
            token = token.lower()                
            if token in stopwords_en:          
                continue
            token = lmtzr.lemmatize(token, 'v')   
            final_token.append(token) 

    return final_token


def join_list_to_string(lst):
    return ' '.join(lst)

