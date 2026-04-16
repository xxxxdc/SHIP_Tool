# 开源环境下隐匿漏洞补丁识别工具平台SHIP



## 文件结构

本项目主要的文件结构如下所示。代码部分包含根据我们的方法SHIP实现的工具，数据部分包含我们所使用的数据集中所有漏洞和补丁的信息。

```bash
├── ship/
│   ├── build/            				# 构建工具配置
│   ├── config/           				# 环境和项目配置
│   ├── dist/             				# 前端静态文件
│   ├── src/                			# 前端页面
│   ├── deepseek.py          			# 调用大模型获取代码变更摘要和是否为补丁的判断结果
│   ├── get_each_commit_feature.py      # Phase3-计算每个代码提交的基于规则的特征
│   ├── get_feature.py      			# Phase1-计算每个代码提交的基于规则的特征
│   ├── group_ranking.py          		# Phase3-对所有提交组进行排序，推荐排名第一的组为补丁组
│   ├── initial_ranking.py              # Phase1-计算每个代码提交的语义特征并进行初始排序
│   ├── interrelationship_feature.py    # Phase2-计算每对代码提交的基于规则的特征
│   ├── manage.py                       # 后端程序
│   ├── predict_relevance_score.py      # Phase2-计算每对代码提交的语义特征并预测代码提交间的相关性分数
│   ├── util.py                       	# 自定义函数文件
│   ├── favicon.ico           			# SHIP logo图片
│   ├── index.html          			# 主页面入口文件
│   ├── package-lock.json          		# 依赖锁定文件
│   ├── package.json          			# 项目配置文件
├── data/
│   ├── multi.csv                     	# 数据集文件 (包含单补丁和多补丁漏洞，共4,630条数据)
│   ├── ghsa_dict.pkl                	# CVE ID和GHSA ID的对应关系文件
│   ├── vuln_type_impact.json        	# 漏洞类型和影响文件 (用于Phase1)
├── requirements.txt       				# 依赖环境配置文件
```



## 如何运行

### 方法一：自行部署

实验在Ubuntu 20.04、cuda 12.2的机器上进行。用户可按照如下步骤进行部署：

1. 将https://figshare.com/articles/dataset/SHIP_models/30162115中提供的三个模型权重参数文件下载至 ship/ 目录 

2. 创建并激活conda虚拟环境

   ```bash
   conda create -n SHIP python=3.9
   conda activate SHIP
   ```

3. 安装 PyTorch, torchaudio, torchvision

   ```bash
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. 从官网下载并安装 nodejs 16.2.0和MySQL

5. 将数据集文件multi.csv导入到MySQL数据库

6. 安装其他依赖库

   ```bash
   pip install -r requirements.txt
   ```

7. 运行后端程序

   ```bash
   python manage.py
   ```

8. 安装前端依赖

   ```bash
   npm install
   ```

9. 运行前端程序

   ```bash
   npm run dev
   ```

   

### 方法二：直接使用Docker

用户可按照如下步骤使用我们提供的Docker镜像：

1. 下载Docker镜像https://figshare.com/articles/dataset/SHIP_docker/30162439

2. 在终端内导入并运行Docker镜像

   ```bash
   docker import ship_web.tar ship_web
   docker run -it --gpus all -p 8080:8080 -p 5000:5000 ship_web /bin/bash
   ```

3. 运行脚本文件以启动服务

   ```bash
   sh /home/start.sh
   ```

   
