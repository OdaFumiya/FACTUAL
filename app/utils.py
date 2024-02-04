import torch
import datetime
import os

START_TAG = '<START>'
STOP_TAG = '<STOP>'
DEVICE = torch.device('cpu')

def delete_zero(tens):
    return tens[tens!=0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long,device=DEVICE)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def getTimeStr():
  t_delta = datetime.timedelta(hours=9)
  JST = datetime.timezone(t_delta, 'JST')
  now = datetime.datetime.now(JST)
  return now.strftime('%Y%m%d%H%M%S')

def getSameExtension(dir,extension):
  ret=[]
  for j in os.listdir(dir):
    if j.endswith(extension):ret.append(j)
  return ret