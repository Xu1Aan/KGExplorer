import pandas as pd

# data template
todos = ['train', 'dev', 'test']
tag_set = ['B_手术',
           'I_疾病和诊断',
           'B_症状',
           'I_解剖部位',
           'I_药物',
           'B_影像检查',
           'B_药物',
           'B_疾病和诊断',
           'I_影像检查',
           'I_手术',
           'B_解剖部位',
           'O',
           'B_实验室检验',
           'I_症状',
           'I_实验室检验']
tag2id = lambda tag: tag_set.index(tag)
id2tag = lambda id: tag_set[id]
dir_path = '../data/chinese_biomedical_NER_dataset'

# build dataset
train_texts = []
train_tags = []
train_ids = []

dev_texts = []
dev_tags = []
dev_ids = []

test_texts = []
test_tags = []
test_ids = []

data = {
    'train': (train_texts, train_tags, train_ids),
    'dev': (dev_texts, dev_tags, dev_ids),
    'test': (test_texts, test_tags, test_ids)
}

text_sequence = []
tag_sequence = []
id_sequence = []


def data_extract():
    for _ in todos:
        path = f'{dir_path}/{_}.txt'
        print(path)

        texts, tags, ids = data[_]

        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line == '\n':
                    texts.append(text_sequence.copy())
                    tags.append(tag_sequence.copy())
                    ids.append(id_sequence.copy())

                    text_sequence.clear()
                    tag_sequence.clear()
                    id_sequence.clear()

                else:
                    token = line[0]
                    tag = line[2:-1]
                    id = tag2id(tag)

                    text_sequence.append(token)
                    tag_sequence.append(tag)
                    id_sequence.append(id)

if __name__ == '__main__':
    build_dataset()
    train_data = pd.DataFrame(list(zip(train_texts, train_tags, train_ids)), columns=['sequences', 'tags', 'tag_ids'])
    dev_data = pd.DataFrame(list(zip(dev_texts, dev_tags, dev_ids)), columns=['sequences', 'tags', 'tag_ids'])
    test_data = pd.DataFrame(list(zip(test_texts, test_tags, test_ids)), columns=['sequences', 'tags', 'tag_ids'])
    train_data.to_csv('train.csv')
    dev_data.to_csv('dev.csv')
    test_data.to_csv('test.csv')
