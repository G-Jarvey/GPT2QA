import json
from tqdm import tqdm
data_path = '../data/raw_data/durecdial.txt'
save_apth = '../data/prepared_data/durecdial_data.txt'
remove = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]']
save_file = open(save_apth, 'a', encoding='utf-8')
with open(data_path, encoding='utf-8') as f:
    dialogues = f.readlines()
    for dialogue in tqdm(dialogues):
        dialogue = json.loads(dialogue)  # 每轮对话
        dialogue = dialogue['conversation']
        # print(dialogue)
        for sentence in dialogue:
            for a in remove:
                sentence = sentence.replace(a, '').replace(' ', '')
            # print(sentence)
            save_file.write(sentence)
            save_file.write('\n')
        save_file.write('\n')

save_file.close()


