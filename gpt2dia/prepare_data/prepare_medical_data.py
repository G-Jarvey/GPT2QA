import pandas as pd
path = '../data/raw_data/妇产科6-28000.csv'
write_file = open('../data/prepared_data/medical_data.txt', 'a', encoding='utf-8')
# 使用pandas读入
data = pd.read_csv(path, encoding='GB18030')
for title, ask, answer in zip(data['title'], data['ask'], data['answer']):
    write_file.write(title.replace(' ', '').replace('\n', '') + ask.replace(' ', '').replace('\n', '') + '\n')
    write_file.write(answer.replace(' ', '').replace('\n', '') + '\n\n')
write_file.close()
print(data['department'][0])