import json


raw_data_path = '../data/raw_data/du_knowledge.txt'
write_file = open('../data/prepared_data/du_knowledge.txt', 'a', encoding='utf-8')

with open(raw_data_path, encoding='utf-8') as f:
    data = f.readlines()
    for dialogue in data:
        dialogue = json.loads(dialogue)['conversation']
        for sentence in dialogue:
            sentence = sentence['utterance'].replace(' ', '')
            write_file.write(sentence)
            write_file.write('\n')
        write_file.write('\n')
write_file.close()


print(json.loads(data[0])['conversation'][8]['utterance'])
