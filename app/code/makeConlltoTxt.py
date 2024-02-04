def load_data_and_labels(filename, encoding='utf-8'):
    lists=[]
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                lists.append((words,tags))
                words, tags = [], []
    return lists
