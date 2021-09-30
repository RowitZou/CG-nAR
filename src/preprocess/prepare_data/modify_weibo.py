import numpy as np

features_1 = np.load('src/preprocess/prepare_data/graph_embedding_temp.npy')

pad = np.zeros([1, len(features_1[0])], dtype='float32')
unk = np.random.randn(1, len(features_1[0]))

add = np.concatenate((pad, unk), axis=0)

features_1 = np.concatenate((features_1, add), axis=0)

np.save('src/preprocess/prepare_data/weibo_graph_embedding.npy', features_1)

modify = open('src/preprocess/prepare_data/vertex.txt', 'w')
modify.write('[pad] 0' + '\n')
modify.write('[UNK] 1' + '\n')

with open('src/preprocess/prepare_data/vertex_temp.txt', 'r') as raw:
    lines = raw.readlines()
    for line in lines:
        line = line.strip().replace('\n', '').split(' ')
        if len(line) < 2:
            continue
        line[1] = str(int(line[1]) + 1)
        write_line = ' '.join(line) + '\n'
        modify.write(write_line)
