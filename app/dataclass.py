from msilib import Feature
import os
import csv
import torch
from filereader import load_data_and_labels
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Prepared_Dataset(Dataset):
    def __init__(self, features, labels,filename):
        super().__init__()
        self.features=features
        self.labels=labels
        self.filename=filename
        self.len=len(self.features)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        feature=self.features[index]
        label=self.labels[index]
        return feature, label
    def get_filename(self):
        return self.filename


class LearningData:
    def __init__(self,traindatafile):
      self.filepath=traindatafile
      self.filename=os.path.basename(traindatafile)
      self.sentence_and_tags = load_data_and_labels(traindatafile)
      self.data=[]
      tmp=[]
      for i in range(len(self.sentence_and_tags[0][0])):
        if(self.sentence_and_tags[0][0][i]!="<DIV>"):
          tmp.append((self.sentence_and_tags[0][0][i],self.sentence_and_tags[0][1][i]))
        else :
          self.data.append(tmp)
          tmp=[]
          
    def checkviolation(self):
      print("a")
    
    def print(self):
      print("filepath="+self.filepath)
      print("filename="+self.filename)
      print(self.data)

    def printstrlist(self):
      for l in self.data:
        tmpstr=""
        for i in l:
          tmpstr+=i[0]
        print(tmpstr)
      print(len(self.data))


    def get_data(self):
      return self.data
    
    def get_sentence_and_tags(self):
      return self.sentence_and_tags

    def get_sentence_and_tags_from_req(self):
      ret=[]
      for l in self.data:
        tmpsentence=[]
        tmptags=[]
        for i in l:
          tmpsentence.append(i[0])
          tmptags.append(i[1])
        ret.append((tmpsentence,tmptags))
      return ret

    def get_filename(self):
      return self.filename


class ReqData:
    def __init__(self,file):
      self.filepath = file
      self.reqsentencelist=self.readtxt(self.filepath)
      self.req = self.parser()

    def readcsv(self,csvfile):
      with open(csvfile, 'w', newline='') as csv:
        spamreader = csv.reader(csv, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(', '.join(row))
    def readtxt(self,txtfile):
      f = open(txtfile, 'r', encoding="utf-8")
      ret = f.read().split('\n')
      f.close
      return ret
      