import json

movie_path = '../data/raw_data/movie_B.txt'
save_path = '../data/prepared_data/movie_data.txt'

write_file = open(save_path, 'a', encoding='utf-8')
with open(movie_path, encoding='utf-8') as f:
    data = f.readlines()
    for dialogue in data:  # 每行是一组对话
        dialogue = json.loads(dialogue)
        for sentence in  dialogue:  # 每组对话的每一句
            write_file.write(sentence['utter'])
            write_file.write('\n')
        write_file.write('\n')
write_file.close()




