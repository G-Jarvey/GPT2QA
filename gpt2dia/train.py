import argparse
import logging
import torch
import random
import numpy as np
import os
import transformers
import pickle
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

from transformers import BertTokenizerFast
from transformers import GPT2Config, GPT2LMHeadModel
from torch.nn import DataParallel
from dataset import MyDataset
from torch.utils.data import DataLoader
from datetime import datetime
from os.path import join
from pytorchtools import EarlyStopping


def setup_train_args():
    # 设置训练参数
    # required - 可选参数是否可以省略
    # action - 出现该参数则设为默认值
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置GPU')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU')
    parser.add_argument('--model_config', default='config/config.json', type=str, required=False, help='模型参数')
    parser.add_argument('--vocab_path', default='vocab/vocab_s.txt', type=str, required=False, help='词库')
    parser.add_argument('--train_data_path', default='data/prepared_data/1w.pkl', type=str, required=False, help='训练数据')
    parser.add_argument('--max_len', default=200, type=int, required=False, help='input数据的最大长度')
    parser.add_argument('--log_path', default='log/train.log', type=str, required=False, help='训练日志')
    parser.add_argument('--tokenized', action='store_true', help='是否对data进行tokenized')
    parser.add_argument('--epochs', default=30, type=int, required=False, help='训练轮数')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='批次大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--warmup', default=500, type=int, required=False, help='热身步数')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步一次loss')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--grad_accumulate', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--model_output_path', default='model/test/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型')
    parser.add_argument('--tensorboard_dir', default='tensorboard_summary/', type=str, required=False,
                        help='tensorboard路径')
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')
    parser.add_argument('--patience', type=int, default=0, help='用于early stopping')
    parser.add_argument('--num_workers', type=int, default=1, help='dataloader加载多线程')
    parser.add_argument('--label_smooth', default=True, action='store_true', help='是否进行标签平滑')
    parser.add_argument('--val_num', type=int, default=1000, help='验证集')

    return parser.parse_args()


def create_logger(args):
    # 输出日志到控制台和文件

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建handler，写入日志文件
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)

    return input_ids, labels


def set_random_seed(args):
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all() # 多块gpu
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        # 当使用新的尺度参数的时候，cuDNN 自动从几种算法里面寻找最适合当前配置的高效算法，之后所有相同参数的数据都采用这个算法。
        # 但是由于噪声等造成即使在同一个机器也可能会选择不同的算法。
        torch.backends.cudnn.benchmark = False  # 复现 // True能加速


def load_dataset(logger, args):
    # 加载训练集和验证集

    logger.info("loading data")
    train_path = args.train_data_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分
    val_num = args.val_num
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]

    train_dataset = MyDataset(input_list_train, args.max_len)
    val_dataset = MyDataset(input_list_val, args.max_len)

    return train_dataset, val_dataset


def calculate_acc(logit, labels, ignore_index=-100):
    a = logit
    b = labels
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))  #contiguous(拷贝，与原数据无关)
    labels = labels[..., 1:].contiguous().view(-1)  #view(重构维度，需要continue)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，将labels中为pad_id的位置置为0
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()  #item(获得内容)
    #########
    shift_logits = a[..., :-1, :].contiguous()
    shift_labels = b[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    not_ignore = shift_labels.ne(ignore_index)  #ne(按照ignore_index划分为True和False)
    num_targets = not_ignore.long().sum().item()  #long(向下取整)
    ##########
    loss /= num_targets
    return n_correct, n_word, loss


def train_one_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args):
    logger.info("start train epoch {}".format(epoch + 1))
    # 训练一个epoch
    model.train()
    device = args.device

    ignore_index = args.ignore_index
    start_time = datetime.now()
    total_loss = 0

    epoch_correct_num, epoch_total_num = 0, 0  # 分别为预测正确的单词数和本次预测的单词总数

    # 循环batch_size的data
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):

        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计一个batch的word数
            batch_correct_num, batch_total_num, a_loss = calculate_acc(logits, labels, ignore_index=ignore_index)

            # 统计epoch的word数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num

            # 计算batch的acc
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.grad_accumulate > 1:
                loss = loss / args.grad_accumulate

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计后，更新参数
            if (batch_idx + 1) % args.grad_accumulate == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info("batch {} of epoch {}, loss {}, acc {}, lr {}, a_loss {}".format(batch_idx + 1, epoch + 1,
                                                                                  loss.item() * args.grad_accumulate,
                                                                                  batch_acc, scheduler.get_lr(), a_loss))

            del input_ids, outputs

        # 处理爆显存问题，要清空gpu占用
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 计算当前epoch的平均loss和acc
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num

    # 保存epoch的信息
    logger.info('epoch {}: loss {}, acc {}'.format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # 每个epoch保存一次model
    logger.info('saving model of epoch {}'.format(epoch + 1))

    model_path = join(args.model_output_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    finish_time = datetime.now()
    logger.info('time of train epoch: {}'.format(finish_time - start_time))

    return epoch_mean_loss


def val_one_epoch(model, val_dataloader, logger, epoch, args):
    logger.info('start val')
    model.eval()
    device = args.device
    start_time = datetime.now()
    total_loss = 0

    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(val_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 当前epoch的平均loss
            epoch_mean_loss = total_loss / len(val_dataloader)
            logger.info('val epoch {}: loss {}'.format(epoch + 1, epoch_mean_loss))
            finish_time = datetime.now()
            logger.info('time of val this epoch: {}'.format(finish_time - start_time))

            return epoch_mean_loss

    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logging.info(str(exception))
            raise exception


def train(model, logger, train_dataset, val_dataset, args):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate_fn, drop_last=True)

    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.model_output_path)
    total_step = len(train_dataloader) // args.grad_accumulate * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                                             num_training_steps=total_step)

    logger.info('start training')

    # 记录每个epoch的loss
    train_loss, val_loss = [], []

    # 验证集最小的loss
    best_val_loss = 100000

    # 开始训练
    for epoch in range(args.epochs):
        # training
        epoch_train_loss = train_one_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                                           scheduler=scheduler, logger=logger, epoch=epoch, args=args)
        train_loss.append(epoch_train_loss)

        # val
        epoch_val_loss = val_one_epoch(model=model, val_dataloader=val_dataloader, logger=logger, epoch=epoch,
                                       args=args)
        val_loss.append(epoch_val_loss)

        # 保存当前在val上loss最低的model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            logger.info('current model of epoch {}, loss {}'.format(epoch + 1, best_val_loss))
            model_path = join(args.model_output_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        # early stopping
        if args.patience == 0:
            continue
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            logger.info('early stopping')
            break

    logger.info('training finish')
    logger.info('train_loss: {}, val_loss: {}'.format(train_loss, val_loss))


def main():
    args = setup_train_args()  # 获取参数

    global logger
    logger = create_logger(args)  # 创建日志

    args.cuda = torch.cuda.is_available() and not args.no_cuda  # cuda可用且参数未设置nocuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 设置随机种子
    if args.seed:
        set_random_seed(args)

    # 配置显卡环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # 使用bert进行编码
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    # 创建对话模型的输出目录
    if not os.path.exists(args.model_output_path):
        os.makedirs(args.model_output_path)

    # 创建model
    if args.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:
        model_config = GPT2Config.from_pretrained(args.model_config)
        model = GPT2LMHeadModel(config=model_config)

    model = model.to(device)

    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 多块gpu并行
    mutli_gpu = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("u are using gpus")
        model = DataParallel(model, device_ids=[int(i) for i in args.device_ids.split(",")])  # 输入参数为 --device 0,1,2
        mutli_gpu = True

    # 记录模型参数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("the number of model parameters: {}".format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    train_dataset, val_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, val_dataset, args)


if __name__ == '__main__':
    main()
