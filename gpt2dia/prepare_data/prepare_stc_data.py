import json
import string

stc_data_path = './raw_data/STC.json'
stc_prepared_data = './data/STC_data.txt'
with open(stc_data_path, encoding='utf-8') as f:
    lists = json.load(f)

print(lists['train'][0])  # ['二 十 六 年 前 的 我 挺 瘦 吧 ？ 不 知 这 几 位 盲 童 现 在 好 吗 ？', '恩 ， 不 但 瘦 ， 头 发 还 很 多 。']

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！＂《》「🔫」—∩😳•﹏•´▽ﾉ【】（）、。：；’‘……￥·"""

dicts = {i: '' for i in punctuation}
punc_table = str.maketrans(dicts)

write_file = open(stc_prepared_data, 'a', encoding='utf-8')
for i in range(len(lists['train'])):  # 所有轮对话
    for j in range(len(lists['train'][i])):  # 每轮对话
        sentence = lists['train'][i][j].replace(" ", "")
        sentence = sentence.translate(punc_table)
        write_file.write(sentence)
        write_file.write('\n')
    write_file.write('\n')
write_file.close()
