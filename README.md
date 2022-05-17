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

## 使用方法

### 数据准备
本实验采用多轮对话数据的输入，使用了公开的[对话语料](https://github.com/codemayq/chinese_chatbot_corpus)以及其他一些公开的数据集（如[千言](https://www.luge.ai/#/)提供的一些），读者可自行编写脚本进行数据的清洗筛选工作，最后应当生成如下形式的数据：


```
喂，吉姆，晚饭后去喝点啤酒怎么样？
你知道这很诱人，但对我们的健康真的不好。
什么意思?它会帮助我们放松。
你真的这么认为吗？我没有。这只会让我们变胖，变傻。记得上次吗？
我想你是对的。但是我们该怎么办呢？我不想坐在家里。
我建议去健身房，在那里我们可以唱歌，还可以认识一些朋友。
这是个好主意。我听说玛丽和莎莉经常去那里打乒乓球。也许我们可以和他们组成一个四人组。
听起来很棒！如果他们愿意，我们可以请他们和我们一起去跳舞。那也是极好的锻炼和乐趣。
很好。我们走吧。
好吧。

你会做俯卧撑吗？
我当然可以。小菜一碟！信不信由你，我一分钟可以做30个俯卧撑。
真的吗？我认为那是不可能的！
你是说30个俯卧撑？
耶！
很简单。如果你每天锻炼，你也能做到。

你能开着收音机学习吗？
不，我听背景音乐。
有什么不同？
收音机里的新闻太多了。
没错，但是你必须买一台电唱机。
```


不同角色对话换行区分，不同批次对话空行区分，然后存储到txt文件中，修改pocess_data.py的相应路径并运行生成对应的pkl文件，同时输出对话长度的均值、中位、最大等参数，以供训练调参，pkl中的数据已经过BertTokenizerFast的编码，不同句子以[sep]隔开。这里给出本次实验可能使用到的一些[数据集，提取码：m9t2](https://pan.baidu.com/s/1vATZN4_SAnQTMnelFLiqhQ) 。

### 模型训练
模型使用了GPT2模型，架构如下：
![image](https://user-images.githubusercontent.com/74944178/168749606-fde0391c-7993-4b50-a493-197c9fb3a1b3.png)
模型的对应输入格式为

```
[CLS]sentence1[SEP]sentence2[SEP]sentence3[SEP]sentence4[SEP]
```

每轮对话输入到模型中进行自回归训练，能够实现生成文本的功能。
修改train.py内的参数，如batch_size、epochs、device等后直接运行，或在终端输入如下形式命令开启训练

```
python train.py --device 0,1 --max_len 300 --epochs 50
```

### 用户交互
本实验共提供两种交互模型，终端+web，其中终端交互修改interact.py文件中加载的模型路径以及生成回复的参数等后直接运行即可，exit为终止符。
web部署的交互部分参考了[chatbot](https://github.com/sylviapap/chatbot)项目，修改对应传参，将用户输入传回本地输入模型，并将模型输出传回web，同时具备语音播报的功能。

        
        
            
            
