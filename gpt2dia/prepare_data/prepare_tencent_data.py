import json
import codecs
dialog_list = json.loads(codecs.open("../data/raw_data/tencent.json", "r", "utf-8").read())
write_file = open('../data/prepared_data/tencent_data.txt', 'a', encoding='utf-8')
for dialogue in dialog_list:
    dialogue = dialogue['content']
    for sentence in dialogue:
        write_file.write(sentence)
        write_file.write('\n')
    write_file.write('\n')
write_file.close()

print(dialog_list[100]['content'])