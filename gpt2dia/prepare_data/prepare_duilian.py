in_path = '../data/raw_data/in.txt'
out_path = '../data/raw_data/out.txt'
write_path = '../data/prepared_data/duilian_data.txt'
write_file = open(write_path, 'a', encoding='utf-8')
with open(in_path, encoding='utf-8') as in_file, open(out_path, encoding='utf-8') as out_file:
    in_data = in_file.readlines()
    out_data = out_file.readlines()
    for up, down in zip(in_data, out_data):
        up, down = up.replace(' ', ''), down.replace(' ', '')
        write_file.write(up)
        write_file.write(down)
        write_file.write('\n')
write_file.close()