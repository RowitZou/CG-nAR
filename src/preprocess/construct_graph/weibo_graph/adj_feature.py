import pickle
import random

graph = {}
adj_list = []
keyword = ''
with open('resource/graph_data/weibo_graph/weibo-graph.txt', 'r') as f_graph:
    lines = f_graph.readlines()
    for line in lines:
        line = line.strip().replace('\n', '')
        if ':' in line:
            if keyword != '' and len(adj_list) > 0:
                graph[keyword] = adj_list
                adj_list = []
            keyword = line[:-1]
        elif len(line) > 1:
            word = line[:]
            if word != keyword:
                adj_list.append(word)

keys = graph.keys()
# print(keys)
for key, values in graph.items():
    for value in values:
        if value not in keys:
            graph[key].remove(value)
            # print(value)
length = len(graph)
# print(length)
vertex2id = {}
adj_matrix = []
index = 0

with open('src/preprocess/prepare_data/vertex.txt', 'w') as vocab:
    for key in keys:
        vertex2id[key] = index
        index += 1
        line = key + ' ' + str(index) + '\n'
        vocab.write(line)

with open('src/preprocess/prepare_data/adj_matrix.txt', 'wb') as adj:
    index = 0
    for key in keys:
        if index != vertex2id[key]:
            print(key + 'is not right !!!')
        adj_list = [0] * length
        for value in graph[key]:
            if value in keys:
                idx = vertex2id[value]
                adj_list[idx] = 1
        adj_matrix.append(adj_list)
        index += 1
    pickle.dump(adj_matrix, adj)

entity_vocab = {}
idx2entity = {}

# with open('./weibo_vertex.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         line = line.strip().split(' ')
#         entity_vocab[line[0]] = int(line[1])
#         idx2entity[int(line[1])] = line[0]

# print(idx2entity[1])

# with open('./adj_matrix.txt', 'rb') as adj:
#     adj_m = pickle.load(adj)

glove = open('resource/cn_bi_fastnlp_100d.txt', 'r')
word = list()
word_vector = list()
glove_dict = {}
line = glove.readline()
index = 0
while line:
    line = list(line.split())
    word.append(line[0])
    new_list = line[1:]
    new_list = list(map(float, new_list))
    word_vector.append(new_list)
    line = glove.readline()
    glove_dict[word[index]] = new_list
    index += 1

feature_matrix = []

for key in keys:
    if key not in word:
        # print(key)
        feature_matrix.append(word_vector[random.randint(0, len(word))])
    else:
        feature_matrix.append(glove_dict[key])

with open('src/preprocess/prepare_data/weibo_features.txt', 'wb') as features:
    pickle.dump(feature_matrix, features)