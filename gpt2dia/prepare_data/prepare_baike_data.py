from tqdm import tqdm
import json

baike_data_path = '../data/raw_data/baike_qa_train.json'
baike_prepared_data = '../data/prepared_data/baike_data.txt'
write_file = open(baike_prepared_data, 'a', encoding='utf-8')
with open(baike_data_path, encoding='utf-8') as f:
    for line in tqdm(f.readlines()):  # 每一行是一组
        sentences = json.loads(line)
        question = sentences['title'] + sentences['desc']
        question = question.replace('\r', '').replace('\n', '').replace(' ', '')
        write_file.write(question)
        write_file.write('\n')
        answer = sentences['answer'].replace('\r', '').replace('\n', '').replace(' ', '')
        write_file.write(answer)
        write_file.write('\n\n')
write_file.close()
