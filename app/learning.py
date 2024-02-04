# Author: Robert Guthrie

from calendar import EPOCH
from model import BiLSTM_CRF
from diffcheck import outputresult
from utils import START_TAG, STOP_TAG,DEVICE, prepare_sequence,delete_zero
from dataclass import LearningData,Prepared_Dataset
from multiprocessing import Process, Queue
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn as rnn
import numpy as np
import multiprocessing as mp
import sys
import os
import gensim
from tqdm import tqdm
import json
import random
args = sys.argv

WORD_TO_IX={"<pad>":0,START_TAG:1,STOP_TAG:2,}
TAG_TO_IX={"<pad>":0,START_TAG:1,STOP_TAG:2,'I-ILF': 3, 'B-ILF': 4, 'I-EIF': 5, 'B-EIF': 6,'I-EI': 7,'B-EI': 8,'I-EQ': 9,'B-EQ': 10, 'I-EO': 11, 'B-EO': 12, 'I-TF': 13, 'B-TF': 14,'I-DF': 15, 'B-DF': 16,'O': 17,}

def matchCheck(ans,predict):
  def convert(tag):
      tmp=-1
      match tag:
        case 'I-ILF':
            tmp=0
        case 'I-EIF':
            tmp=1
        case 'I-EI':
            tmp=2
        case 'I-EQ':
            tmp=3
        case 'I-EO':
            tmp=4
        case 'I-TS':
            tmp=5
        case 'I-ES':
            tmp=6
        case 'I-TF':
            tmp=7
        case 'I-DF':
            tmp=8
        case 'I-item':
            tmp=8
        case _:
            tmp=-1
      return tmp
  
  ret=[False,False,len(ans)]
  prediction_function_num=[0,0,0,0,0,0,0,0,0,0]
  ans_function_num=[0,0,0,0,0,0,0,0,0,0]
  if(ans==predict):
    ret=[True,True,len(ans)]
  else:
    correct_num=0
    for i in range(len(predict)):
      tmp=convert(predict[i])
      if(tmp!=-1):prediction_function_num[tmp]+=1
      tmp=convert(ans[i])
      if(tmp!=-1):ans_function_num[tmp]+=1
      if(predict[i]==ans[i]):correct_num+=1
    if(prediction_function_num==ans_function_num):ret[0]=True
    ret[2]=correct_num
  return ret

def makedic(model,x,y):
    loss,result_list = model(delete_zero(x))
    sentence=[]
    ans=[]
    tag_list=[]
    for i in range(len(result_list)):
      tag_list.append(list(TAG_TO_IX)[result_list[i]])
      sentence.append(list(WORD_TO_IX)[x[i]])
      ans.append(list(TAG_TO_IX)[y[i]])
    match=matchCheck(ans,tag_list)
    return{"sentence":sentence,"ans":ans,"predict":tag_list,"correct_require":match[0],"perfect_matching_require_num":match[1],"correct_word_num":match[2],"word_num":len(delete_zero(x)),"loss":loss.item()}

def outputResultJSON(model, dataloader,dirname,filename):
    with torch.no_grad():
      ret={}
      for bat in tqdm(dataloader, desc="result:batch", leave=False):        
        d=torch.stack(bat,dim=1).to(DEVICE)
        for x,y in tqdm(d, desc="result:req", leave=False):
          ret[len(ret)]=makedic(model,x,y)
    f=os.path.basename(filename).split('.', 1)[0]
    with open(f'{dirname}{f}.json', 'w',newline="",encoding="utf-8") as f:
      json.dump(ret,f,indent=2, sort_keys=True, ensure_ascii=False, separators=(',', ': '))
      f.close()



def outputResult(resultdir,num,model,train_data_loader,traindatafilename,model_save_flag=True):
  if not os.path.exists(f"{resultdir}\\{num}"):
    os.makedirs(f"{resultdir}\\{num}")
  dirname=f"{resultdir}\\{num}\\"
  
  if(model_save_flag):model.save_model(f".\\{resultdir}\\model.pth")
  outputResultJSON(model, train_data_loader,dirname,traindatafilename)
  with open(f".\\{dirname}word_to_ix.txt", 'w', encoding="UTF-8", newline="") as f:
    print(WORD_TO_IX, file=f)
  with open(f".\\{dirname}tag_to_ix.txt", 'w', encoding="UTF-8", newline="") as f:
    print(TAG_TO_IX, file=f)


def dataFileName(resultdir,num,path):
  return f'{resultdir}\\{num}\\{path}'

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


def learning(traindatafilename, resultdir="models", epoch=0,interval=-1,bat=16):
    global WORD_TO_IX
    global TAG_TO_IX
    debug_flag=False
    HIDDEN_DIM = 32
    EPOCH_NUM = epoch
    bat_size=bat
    WORD_TO_VECTOR_MODEL_PATH = ".\system_file\size200-min_count20-window10-sg1\wikipedia.model"
    # 結果出力ファイル作成
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    # グラボの準備
    if torch.cuda.is_available():
      #torch.set_default_tensor_type(torch.cuda.FloatTensor)
      torch.backends.cudnn.benchmark = True
    
    # 学習/試験用データ読み込み
    train_data=LearningData(traindatafilename)
    makeDatatoIx(train_data.get_sentence_and_tags())


    # 学習用Dataloader

    max_length=-1
    for t in tqdm((train_data.get_sentence_and_tags_from_req()), desc="req", leave=False):
      for sentence, tags in [t]:
        if(len(sentence)>max_length):max_length=len(sentence)
    
      
    tmp_features=[]
    tmp_train_labels=[]
    for t in tqdm((train_data.get_sentence_and_tags_from_req()), desc="req", leave=False):
      for sentence, tags in [t]:
        sentence_in = prepare_sequence(sentence, WORD_TO_IX)
        targets = torch.tensor([TAG_TO_IX[t]for t in tags], dtype=torch.long)
        tmp_features.append(sentence_in)
        tmp_train_labels.append(targets)
    train_features=rnn.pad_sequence(tmp_features, batch_first=True).to('cpu')
    train_labels=rnn.pad_sequence(tmp_train_labels, batch_first=True).to('cpu')
    tmp_features=[]
    tmp_train_labels=[]


    train_dataset = Prepared_Dataset(train_features, train_labels,traindatafilename)
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bat_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4, 
        )
    valid_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=False,
        num_workers=4, 
        )


    #モデル初期化

    model = BiLSTM_CRF(len(WORD_TO_IX), TAG_TO_IX,
                       HIDDEN_DIM, WORD_TO_VECTOR_MODEL_PATH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    torch.backends.cudnn.benchmark = True
    
    for epoch in tqdm(range(EPOCH_NUM), desc="epoch", leave=False):
        model.eval()
        if(interval!=-1 and epoch%interval==0):
          outputResult(resultdir,epoch,model,valid_dataloader,traindatafilename,False)
        model.train()  # 訓練モードに
        for d in tqdm((train_dataloader), desc="learning:batch", leave=False):
          model.zero_grad()
          loss = model.neg_log_likelihood(d).to(DEVICE)
          loss.backward()
          optimizer.step()
    model.eval()
    outputResult(resultdir,epoch,model,valid_dataloader,traindatafilename,True)
    return TAG_TO_IX,WORD_TO_IX
    
def makemodelfromgensim():
    # FloatTensor containing pretrained weights
    weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    embedding = nn.Embedding.from_pretrained(weight)
    # Get embeddings for index 1
    input = torch.LongTensor([1])
    embedding(input)
    model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    # formerly syn0, which is soon deprecated
    weights = torch.FloatTensor(model.vectors)