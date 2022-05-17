import logging
import argparse
import pickle
import joblib
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizerFast
import sys
sys.path.append("..")


def create_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def pre_process():
    # 对原始数据进行tokenize并添加上[CLS],[SEP]
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", default='../vocab/vocab_s.txt', type=str, required=False, help='词表')
    parser.add_argument('--log_path', default='../log/process.log', type=str, required=False, help='日志')
    parser.add_argument('--raw_data_path', default='../data/prepared_data/1w.txt', type=str, required=False, help='原始数据')
    parser.add_argument('--tokenize_data_path', default='../data/token_data/1w.pkl', type=str, required=False,
                        help='tokenize的数据')

    args = parser.parse_args()

    logger = create_logger(args.log_path)

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    logger.info("preprocess data {} to {}".format(args.raw_data_path, args.tokenize_data_path))

    # 读取数据集
    with open(args.raw_data_path, 'rb') as f:
        data = f.read().decode("utf-8")

    # 换行
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")

    logger.info("there are {} dialogue in dataset".format(len(train_data)))

    # 进行tokenize
    dia_len = []  # 记录所有对话tokenize后的长度，用于统计中位数与均值
    dia_list = []

    with open(args.tokenize_data_path, 'w', encoding='utf-8') as f:
        for index, dia in enumerate(tqdm(train_data)):

            # 获取每组对话
            if "\r\n" in data:
                sentences = dia.split("\r\n")
            else:
                sentences = dia.split("\n")

            input_ids = [cls_id]  # 句子以cls开头
            for sentence in sentences:
                input_ids += tokenizer.encode(sentence, add_special_tokens=False)  # 每句话进行编码
                input_ids.append(sep_id)  # 句子间加上sep

            dia_len.append(len(input_ids))  # 加上当前组对话的word长
            dia_list.append(input_ids)  # 将每组对话合并成一句token

    len_mean = np.mean(dia_len)
    len_median = np.median(dia_len)
    len_max = np.max(dia_len)

    # print(dia_list[0])
    # print(dia_list[100])
    with open(args.tokenize_data_path, 'wb') as f:
        joblib.dump(dia_list, f)

    logger.info(
        'process data done, mean of dia len:{}, median of dia len:{}, max of dia len:{}'.format(len_mean, len_median,
                                                                                                len_max))


if __name__ == '__main__':
    pre_process()
