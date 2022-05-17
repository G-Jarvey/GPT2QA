zhdd_data_path = '../data/raw_data/data_zhdd_lines.txt'
write_path = '../data/prepared_data/zhdd_data.txt'

with open(zhdd_data_path, encoding='utf-8') as f:
    data_list = f.readlines()

write_file = open(write_path, 'a', encoding='utf-8')
for dialogue in data_list:  # 一轮对话
    dialogue = dialogue.replace('\n', '').split(' ')
    for sentence in dialogue:
        write_file.write(sentence)
        write_file.write('\n')
    write_file.write('\n')
write_file.close()

print(data_list[1000].replace('\n', '').split(' '))
