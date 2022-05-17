import os
import torch
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify
from interact import Generate
from transformers import BertTokenizerFast
# 需要在不同目录import
import sys
'''
linux:
export FLASK_APP=interact_web
nohup flask run --host=0.0.0.0 -p 5000 > ./log/flask.log

windows:
set FLASK_APP=interact_web
flask run --host=0.0.0.0 -p 5000 > ./log/flask.log
'''
app = Flask(__name__)
CORS(app, resources=r'/*')

history = []
tokenizer = BertTokenizerFast(vocab_file='vocab/vocab_s.txt', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
engine = Generate()
dialogue_file = open('./dialogue/dialogue.txt', 'a', encoding='utf-8')
dialogue_file.write('聊天记录{}:\n'.format(datetime.now()))


@app.route("/", methods=["GET", 'POST'])
def hello_world():
    if request.method == 'POST':
        print('文本生成暂不支持图片和视频请求')
        return

    else:  # GET方法
        question = request.args.get('question')
        industry = request.args.get("industry")

        global history
        history.append(tokenizer.encode(question))

        dialogue_file.write("用户：{}\n".format(question))
        sentences = engine.generate(history=history)
        answer = tokenizer.convert_ids_to_tokens(sentences)
        answer = ''.join(answer)
        if question == 'exit':
            answer = ['请开始下一个话题']
            dialogue_file.write('\n')
            history = []
        else:
            history.append(sentences)
            dialogue_file.write("机器人：{}\n".format(answer))

        return jsonify(return_answer=answer)
