import argparse
import logging
import torch
import os
import torch.nn.functional as F

from datetime import datetime
from transformers import BertTokenizerFast, GPT2LMHeadModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        # torch.topk()返回最后一维最大的k个，形式为(values,indices)
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # 增加一个维度
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulate_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulate_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


class Generate:
    save_dialogue = 'dialogue/'
    model_type = 'test on medical_data_model/epoch10'

    def __init__(self) -> None:
        self.top_k = 10
        self.top_p = 0.5
        self.max_len = 300
        self.repeat_penalty = 2.0
        self.temperature = 1.0
        self.max_history_len = 1
        self.vocab_path = './vocab/vocab_s.txt'
        self.model_path = './model/medical_data_model/epoch10'
        self.tokenizer = BertTokenizerFast(vocab_file=self.vocab_path, sep_token="[SEP]", pad_token="[PAD]",
                                           cls_token="[CLS]")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)

    # 根据当前多轮对话生成下一句话
    def generate(self, history):
        device = 'cuda'
        model = self.model.to('cuda')
        model.eval()

        # 每段话开头为[CLS]
        input_ids = [self.tokenizer.cls_token_id]

        # 最后5句
        for history_id, history_dia in enumerate(history[-self.max_history_len:]):
            input_ids.extend(history_dia)
            input_ids.append(self.tokenizer.sep_token_id)
        now_input_tensor = torch.tensor(input_ids).long().to(device)
        sentences = []

        # 句子裁剪
        for _ in range(self.max_len):
            outputs = self.model(input_ids=now_input_tensor)
            next_token_logits = outputs[0][-1, :]

            # 惩罚生成结果中的重复项
            for id in set(sentences):
                # 概率减小
                next_token_logits[id] /= self.repeat_penalty
            next_token_logits /= self.temperature

            # 将[UNK]的概率设为无穷小，即生成结果不会有nuk
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.top_k, top_p=self.top_p)

            # 从候选集合中无放回抽取num_samples个元素，权重越高抽到的概率越高，返回下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 遇到[SEP]则结束
            if next_token == self.tokenizer.sep_token_id:
                break
            sentences.append(next_token.item())
            now_input_tensor = torch.cat((now_input_tensor, next_token), dim=0)

        return sentences


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(Generate.save_dialogue):
        os.makedirs(Generate.save_dialogue)

    dialogue_file = open(Generate.save_dialogue + '/dialogue.txt', 'a', encoding='utf-8')
    dialogue_file.write('聊天记录{}:\n'.format(datetime.now()) + Generate.model_type + '\n')
    history = []
    vocab_path = 'vocab/vocab_l.txt'
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]",
                                  cls_token="[CLS]")

    print('开始聊天，输入exit退出')
    engine = Generate()
    while True:
        question = input('用户：')
        dialogue_file.write('用户：{}\n'.format(question))
        history.append(tokenizer.encode(question))
        if question == 'exit':
            dialogue_file.write('\n\n')
            dialogue_file.close()
            break
        sentences = engine.generate(history=history)
        history.append(sentences)
        answer = tokenizer.convert_ids_to_tokens(sentences)
        print('机器人：' + ''.join(answer))
        dialogue_file.write('机器人：{}\n'.format(''.join(answer)))
