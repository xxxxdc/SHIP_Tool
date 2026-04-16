import pandas as pd
import os
import re
from openai import OpenAI, OpenAIError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time


def generate_deepseek_analysis(cve_name, api_key="your key",
                              base_url="https://api.deepseek.com", max_workers=25,
                              input_file_suffix="_top50.csv", cache_dir="../cache"):
    
    print(f"开始处理CVE: {cve_name}")
    
    # 读取输入数据
    input_file = f'{cache_dir}/{cve_name}{input_file_suffix}'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    new_df = pd.read_csv(input_file)

    # 数据预处理
    new_df['msg_text'] = new_df['msg_text'].fillna('')
    new_df['diff_code'] = new_df['diff_code'].fillna('')
    new_df['cwe'] = new_df['cwe'].apply(eval)
    new_df['cwe'] = new_df['cwe'].apply(lambda x: [t[1] for t in x] if x else [])
    new_df['cwe'] = new_df['cwe'].apply(lambda x: [] if all(item == 'Other' for item in x) else x)

    # 生成prompt
    for index, row in new_df.iterrows():
        diff_code = row['diff_code']
        if len(diff_code) > 50000:
            diff_code = diff_code[:50000]

        if len(row['cwe']) == 0:
            user_prompt = f"Role: You are a security analyst with expertise in analyzing code changes for functionality and intent, as well as identifying whether those changes address known vulnerabilities or security issues.\n\nYou will be provided with the following information: CVE ID, CVE description (including details about the vulnerability, such as the type of vulnerability, the file or function where it occurs, and its cause), Commit message (the description of the commit), and Commit diff code (the specific code changes in the commit).\n\nFirst, analyze the code changes in the commit diff code, summarize the main changes, and identify any potential vulnerabilities that may be addressed. Common vulnerability types include, but are not limited to: buffer overflow, directory traversal, priviege escalation. If the commit modifies security-relevant code (e.g., memory management, input validation, authentication logic), analyze whether it strengthens security or fixes an existing vulnerability.\n\nSecond, based on your analysis, compare the commit message, code changes and the potential vulnerabilities it addresses with the CVE description to determine whether this commit is a patch for the CVE.\n\nAdditionally, at the end of your response, please summarize the following:\n1. A brief summary of the commit\n2. A list of potential vulnerability types addressed by the commit\n3. Whether this commit is a patch for the vulnerability (YES, NO, or UNKNOWN)\n\nPlease output the results in JSON format as shown below:\n{{\n  \"summarization\": \"summary of the commit\",\n  \"potential addressed vulnerability types\": [\"buffer overflow\", \"directory traversal\", ...],\n  \"is_patch\": \"YES/NO/UNKNOWN\"\n}}\n\nCVE ID:\n{row['cve']}\n\nCVE description:\n{row['desc']}\n\ncommit message:\n{row['msg_text']}\n\ncommit diff code:\n{diff_code}"
        elif len(row['cwe']) == 1:
            user_prompt = f"Role: You are a security analyst with expertise in analyzing code changes for functionality and intent, as well as identifying whether those changes address known vulnerabilities or security issues.\n\nYou will be provided with the following information: CVE ID, CVE description (including details about the vulnerability, such as the type of vulnerability, the file or function where it occurs, and its cause), Commit message (the description of the commit), and Commit diff code (the specific code changes in the commit).\n\nFirst, analyze the code changes in the commit diff code, summarize the main changes, and identify any potential vulnerabilities that may be addressed. Common vulnerability types include, but are not limited to: buffer overflow, directory traversal, priviege escalation. If the commit modifies security-relevant code (e.g., memory management, input validation, authentication logic), analyze whether it strengthens security or fixes an existing vulnerability.\n\nSecond, based on your analysis, compare the commit message, code changes and the potential vulnerabilities it addresses with the CVE description to determine whether this commit is a patch for the CVE.\n\nAdditionally, at the end of your response, please summarize the following:\n1. A brief summary of the commit\n2. A list of potential vulnerability types addressed by the commit\n3. Whether this commit is a patch for the vulnerability (YES, NO, or UNKNOWN)\n\nPlease output the results in JSON format as shown below:\n{{\n  \"summarization\": \"summary of the commit\",\n  \"potential addressed vulnerability types\": [\"buffer overflow\", \"directory traversal\", ...],\n  \"is_patch\": \"YES/NO/UNKNOWN\"\n}}\n\nCVE ID:\n{row['cve']}\n\nCVE description:\n{row['desc']} Its vulnerability type:{row['cwe'][0]}\n\ncommit message:\n{row['msg_text']}\n\ncommit diff code:\n{diff_code}"
        elif len(row['cwe']) > 1:
            result = ', '.join(row['cwe'])
            user_prompt = f"Role: You are a security analyst with expertise in analyzing code changes for functionality and intent, as well as identifying whether those changes address known vulnerabilities or security issues.\n\nYou will be provided with the following information: CVE ID, CVE description (including details about the vulnerability, such as the type of vulnerability, the file or function where it occurs, and its cause), Commit message (the description of the commit), and Commit diff code (the specific code changes in the commit).\n\nFirst, analyze the code changes in the commit diff code, summarize the main changes, and identify any potential vulnerabilities that may be addressed. Common vulnerability types include, but are not limited to: buffer overflow, directory traversal, priviege escalation. If the commit modifies security-relevant code (e.g., memory management, input validation, authentication logic), analyze whether it strengthens security or fixes an existing vulnerability.\n\nSecond, based on your analysis, compare the commit message, code changes and the potential vulnerabilities it addresses with the CVE description to determine whether this commit is a patch for the CVE. \n\nAdditionally, at the end of your response, please summarize the following:\n1. A brief summary of the commit\n2. A list of potential vulnerability types addressed by the commit\n3. Whether this commit is a patch for the vulnerability (YES, NO, or UNKNOWN)\n\nPlease output the results in JSON format as shown below:\n{{\n  \"summarization\": \"summary of the commit\",\n  \"potential addressed vulnerability types\": [\"buffer overflow\", \"directory traversal\", ...],\n  \"is_patch\": \"YES/NO/UNKNOWN\"\n}}\n\nCVE ID:\n{row['cve']}\n\nCVE description:\n{row['desc']} Its vulnerability type:{result}\n\ncommit message:\n{row['msg_text']}\n\ncommit diff code:\n{diff_code}"

        new_df.at[index, 'prompt'] = user_prompt

    # 保存prompt文件
    new_df = new_df[['cve', 'commit', 'prompt']]

    # 设置OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=200
    )

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # 重试间隔时间（秒）

    def process_row(index, row):
        prompt = row.prompt
        if len(prompt) > 64000:
            prompt = prompt[:64000]
        
        retries = 0
        completion = ''
        
        while retries < MAX_RETRIES:
            try:
                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    store=True,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                break
            except OpenAIError as e:
                print(f"调用 API 时第 {index} 行出错: {e}，重试第 {retries + 1} 次...")
                retries += 1
                time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"第 {index} 行出现非 API 相关异常: {e}，不再重试")
                break
        else:
            print(f"第 {index} 行请求重试 {MAX_RETRIES} 次后仍然失败")
            return index, False, ''

        if completion == '':
            return index, False, ''

        text = completion.choices[0].message.content
        return index, True, text

    def parse_json_response(text):
        """解析JSON响应"""
        try:
            # 尝试多种JSON提取模式
            patterns = [
                r'({.*?})',  # 基本模式
                r'```json\s*({.*?})\s*```',  # markdown代码块
                r'```\s*({.*?})\s*```',  # 普通代码块
                r'(\{[^{}]*"summarization"[^{}]*\})',  # 包含summarization的JSON
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    return data
            
            # 如果没有找到JSON格式，尝试直接解析整个文本
            if text.strip().startswith('{') and text.strip().endswith('}'):
                data = json.loads(text.strip())
                return data
                
        except (json.JSONDecodeError, AttributeError):
            return None
        
        return None

    def process_row_with_json_retry(index, row, max_api_retries=3):
        """处理单行数据，如果JSON解析失败则重新调用API"""
        for attempt in range(max_api_retries):
            # 调用API获取响应
            api_index, success, text = process_row(index, row)
            
            if not success:
                print(f"第 {index} 行API调用失败，尝试 {attempt + 1}/{max_api_retries}")
                if attempt < max_api_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return index, False, '', None
            
            # 尝试解析JSON
            parsed_data = parse_json_response(text)
            if parsed_data:
                return index, True, text, parsed_data
            else:
                print(f"第 {index} 行JSON解析失败，重新调用API (尝试 {attempt + 1}/{max_api_retries})")
                if attempt < max_api_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                    continue
        
        # 所有重试都失败了
        print(f"第 {index} 行经过 {max_api_retries} 次API调用仍无法获得有效JSON响应")
        return index, False, text if 'text' in locals() else '', None


    # 直接使用最终输出文件
    final_output = f'{cache_dir}/{cve_name}-deepseek.csv'

    # 检查是否已有结果文件，如果没有则创建
    if not os.path.exists(final_output):
        df_result = new_df[['cve', 'commit']].copy()
        df_result['answer'] = ''
        df_result['summarization'] = ''
        df_result['potential_addressed_vulnerability_types'] = ''
        df_result['is_patch'] = ''
        df_result.to_csv(final_output, index=False)

    # 读取已有的结果文件
    df_result = pd.read_csv(final_output)
    df_result['summarization'] = df_result['summarization'].astype('string')
    df_result['potential_addressed_vulnerability_types'] = df_result['potential_addressed_vulnerability_types'].astype('string')
    df_result['is_patch'] = df_result['is_patch'].astype('string')
    df_result['answer'] = df_result['answer'].astype('string')

    # 将prompt数据与结果数据合并
    df_prompt = new_df
    df_combined = df_result.merge(df_prompt, on=['cve', 'commit'], how='left')

    # 找出需要处理的行（answer列为空）
    indices_to_process = []
    for index, row in df_combined.iterrows():
        if pd.isna(row['answer']) or row['answer'] == '':
            indices_to_process.append(index)

    print(f'总共有 {len(df_combined)} 行数据')
    print(f'需要处理 {len(indices_to_process)} 行数据')

    if len(indices_to_process) > 0:
        print('开始调用DeepSeek API...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        # 一次性处理所有数据，无需分批
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row_with_json_retry, index, df_combined.loc[index]) for index in indices_to_process]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing all data"):
                index, success, text, parsed_data = future.result()
                if success and parsed_data:
                    df_combined.at[index, 'answer'] = text
                    df_combined.at[index, 'summarization'] = parsed_data.get('summarization', '')
                    df_combined.at[index, 'potential_addressed_vulnerability_types'] = str(parsed_data.get('potential addressed vulnerability types', []))
                    df_combined.at[index, 'is_patch'] = parsed_data.get('is_patch', '')
                elif success:
                    # API调用成功但JSON解析失败，仍保存原始响应
                    df_combined.at[index, 'answer'] = text
                    print(f"第 {index} 行JSON解析失败，但原始响应已保存")
                else:
                    print(f"第 {index} 行处理完全失败")
        
        # 处理完成后保存
        result_cols = ['cve', 'commit', 'answer', 'summarization', 'potential_addressed_vulnerability_types', 'is_patch']
        df_combined[result_cols].to_csv(final_output, index=False)

    print(f'所有数据处理完成，结果已保存到: {final_output}')
    
    # 返回处理结果
    result_cols = ['cve', 'commit', 'answer', 'summarization', 'potential_addressed_vulnerability_types', 'is_patch']
    return df_combined[result_cols]


# 主函数调用示例
if __name__ == "__main__":
    cve_name = "CVE-2018-6596"
    result_df = generate_deepseek_analysis(cve_name)
    print("分析完成！")

