# 基于深度学习的自动问答系统（毕业设计）



## 项目介绍
项目参考了[yangjianxin](https://github.com/yangjianxin1/GPT2-chitchat)的项目，总体基于HuggingFace的GPT2模型训练了在开放领域对话数据集和某些限定域的对话数据集上的模型，首先是基于原作者给出的100w数据集以及个人收集到的能用于本设计的数据集来进行预训练模型的训练，再在一些公开的对联、寻医问药数据集（满足正常上下文对话的数据集）上进行类似微调，最终部署在web上完成本项目。

## 运行环境
python==3.8
pytorch==1.5.1
transformer==4.18.0

## 项目结构
```
    |-- gpt2dia
        |-- config
            |-- config.json #模型的参数
        |-- data
            |-- prepared_data #处理好格式的数据，即按照轮次划分
            |-- raw_data #原始数据
            |-- token_data #经过tokenize编码的数据，存到list再存成pkl文件
        |-- dialogue
            |-- dialogue.txt #对话结果保存
        |-- log
            |-- flask.log #flask部署时控制台的信息
            |-- process.log #tokenize文本时的日志，包括数据的length等概况
            |-- train.log #训练时控制台输出，包括当前epoch及acc、loss等
        |-- model #保存训练模型
        |-- prepare_data
            |-- pocess_data.py #进行编码保存到pkl文件中
            |-- xx.py #用于处理收集到的数据，包括提取出对话部分及调整为所需格式（包括一些分段、换行、去除无关字符等操作，脚本自己编写）
        |-- vocab
            |-- vocab.txt #有大小不同两个词表，按需使用
        |-- web_data #web设计的一些文件
        |-- dataset.py #将输入data转tensor，限制长度等操作
        |-- interact.py #终端交互
        |-- interact_web.py #web交互
        |-- train.py #训练
```
        
        
            
            
