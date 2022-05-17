import json

douban_data_path = './raw_data/douban_train.txt'
douban_prepared_data = './data/douban_data.txt'
with open(douban_data_path, encoding='utf-8') as f:
    douban_data = f.readlines()
lis = []
for i in range(len(douban_data)):
    c = json.loads(douban_data[i])
    lis.append(c)  # lis存了所有的数据
write_file = open(douban_prepared_data, 'a', encoding='utf-8')
for i in range(len(lis)):
    for j in range(len(lis[i]['history'])):  # 历史对话有多行
        sentence = lis[i]['history'][j].replace(" ", "")
        write_file.write(sentence)
        write_file.write('\n')
    sentence = lis[i]['response'].replace(" ", "")
    write_file.writelines(sentence)
    write_file.write('\n\n')
write_file.close()
