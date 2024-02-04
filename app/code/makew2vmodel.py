import codecs
import glob
import logging
import os
import pickle
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

word2sid = {"<unk>": "0"} # 単語辞書
files = glob.glob("contents/wiki*")
for fname in tqdm(files):
    for content in codecs.open(fname, 'r', 'utf-8').read().splitlines():
        for token in content.split():
            if token in word2sid:
                continue
            word2sid[token] = str(len(word2sid))
with open("dictionary.pickle", mode="wb") as f:
    pickle.dump(word2sid, f)
sentences = []
for fname in tqdm(files):
    for content in codecs.open(fname, 'r', 'utf-8').read().splitlines():
        sentences.append([word2sid[token] for token in content.split()])
size = 200
min_count = 20
window = 10
sg = 1
dirname = "size{}-min_count{}-window{}-sg{}".format(size, min_count, window, sg)
if not "model/"+dirname in glob.glob("model/*"):
    os.makedirs("model/"+dirname, exist_ok=True)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# モデルを作る
model = Word2Vec(sentences, vector_size=size, min_count=min_count, window=window, sg=sg, epochs=5, workers=10)

# モデルの保存
model.save("model/"+dirname+'/wikipedia.model')