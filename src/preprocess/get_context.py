from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def word_lemmatizer(sentence):
    # get pos tags
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    # tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(sentence)

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return lemmas_sent


def get_context_data(source_file, vertex_list):
    with open(source_file, 'r') as f:
        lines = f.readlines()
    context_file = source_file.replace('source.txt', 'context.txt')
    file = open(context_file, 'w')
    for line in lines:
        write_list = []
        line = line.strip().split('|||')[-1].split(' ')
        line = word_lemmatizer(line)
        for word in line:
            if word in vertex_list:
                write_list.append(word)
        write_line = ' '.join(write_list) + '\n'
        file.write(write_line)


def get_context_list(source_file, vertex_list):
    with open(source_file, 'r') as f:
        lines = f.readlines()
    context_file = source_file.replace('source.txt', 'context_list.txt')
    file = open(context_file, 'w')
    for line in lines:
        write_list = []
        line = line.strip().split('|||')
        for sub_line in line:
            sub_list = []
            sub_line = word_lemmatizer(sub_line.split(' '))
            for word in sub_line:
                if word in vertex_list:
                    sub_list.append(word)
            sub_line = ' '.join(sub_list)
            write_list.append(sub_line)
        write_line = ','.join(write_list) + '\n'
        file.write(write_line)


if __name__ == '__main__':
    types = ['test', 'valid', 'train']
    vertex_list = []
    with open('src/preprocess/prepare_data/vertex.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            vertex = line.strip().split(' ')[0]
            vertex_list.append(vertex)
    for type in types:
        source_file = './tx_data/' + type + '/' + 'source.txt'
        get_context_data(source_file, vertex_list)
        get_context_list(source_file, vertex_list)
        print(type + ' finished!')
