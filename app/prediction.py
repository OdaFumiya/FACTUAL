# Author: Robert Guthrie
import MeCab
from calendar import EPOCH
from model import BiLSTM_CRF
from utils import START_TAG, STOP_TAG, prepare_sequence,getTimeStr
from dataclass import LearningData,ReqData
import torch
import sys
import csv
import os
import json
args = sys.argv

WORD_TO_IX={}
TAG_TO_IX={START_TAG: 0, STOP_TAG: 1}


def makeDatatoIx(data):
    global WORD_TO_IX
    global TAG_TO_IX
    for sentence, tags in data:
      for word in sentence:
          if word not in WORD_TO_IX:
              WORD_TO_IX[word] = len(WORD_TO_IX)
      for tag in tags:
          if tag not in TAG_TO_IX:
              TAG_TO_IX[tag] = len(TAG_TO_IX)
     
def loadmodel(modelfile,hidden,wix,tix):
    word_to_ix = wix
    tag_to_ix = tix
    WORD_TO_VECTOR_MODEL_PATH = ".\system_file\size200-min_count20-window10-sg1\wikipedia.model"

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, hidden, WORD_TO_VECTOR_MODEL_PATH)
    model.load_state_dict(torch.load(modelfile))
    return model

def predictionFromModel(modelfilename,modeldatafilename,reqfilenamelist=[], resultdir=".\\prediction\\"):
    if torch.cuda.is_available():
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
      print(f"USING:{torch.cuda.get_device_name()}")
    global WORD_TO_IX
    global TAG_TO_IX
    debug_flag=False
    HIDDEN_DIM = 32
    # 結果出力ファイル作成
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    jsonfile_open = open(modeldatafilename,"r", encoding="UTF-8")
    jsonfile_load = json.load(jsonfile_open)
    WORD_TO_IX=jsonfile_load["word_to_ix"]
    TAG_TO_IX=jsonfile_load["tag_to_ix"]
    reqdatalist=[]
    reqwordlist=[]
    resultlist=[]

    for reqfilename in reqfilenamelist:
      with open(reqfilename, encoding="UTF-8") as f:
        for line in f:
          reqdatalist.append(line)
    tagger = MeCab.Tagger()  # 「tagger = MeCab.Tagger('-d ' + unidic.DICDIR)」

    for req in reqdatalist:
      tmpwordlist=[]
      p = tagger.parse(req)
      result = p.split('\n')
      for i in range(len(result)):
        word=(result[i].split('\t'))[0]
        if(word=="EOS"):break
        tmpwordlist.append(word)
        if word not in WORD_TO_IX:
          WORD_TO_IX[word] = len(WORD_TO_IX)
      if(len(tmpwordlist)!=0):
        reqwordlist.append(tmpwordlist)
    # グラボの準備
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = loadmodel(modelfilename,HIDDEN_DIM,WORD_TO_IX,TAG_TO_IX)
    for req in reqwordlist:
      wordlist=req
      print(wordlist)
      precheck_sent = prepare_sequence(wordlist, WORD_TO_IX)
      result_list = model(precheck_sent)[1]
      out=[]
      for j in range(len(result_list)):
        out.append([wordlist[j],(list(TAG_TO_IX)[result_list[j]])])
      resultlist.append(out)
    
    filename=os.path.basename(reqfilename).split('.', 1)[0]
    with open(resultdir+filename+'.csv', 'w', encoding="UTF-8",newline="") as f:
      writer = csv.writer(f)
      for r in resultlist:
        writer.writerows(r)
        writer.write("<DIV>,<DIV>")
    