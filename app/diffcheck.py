from filereader import load_data_and_labels
import csv
import os

def diffcheck(datafile,resultfile):
   data=load_data_and_labels(datafile)
   result_data=load_data_and_labels(resultfile)
   same_cnt=0
   O_num=0
   for i in range(len(data[0][0])):
    if(data[0][1][i]!='O'):
      if(data[0][1][i]==result_data[0][1][i]):same_cnt+=1
    else :O_num+=1
   print(same_cnt,len(data[0][0])-O_num)

def outputresult(datafile,resultfile,outputfilename):
  data=datafile
  result_data=load_data_and_labels(resultfile)
  f = open(f'{outputfilename}', 'w', encoding='utf-8', newline='')
  dataWriter = csv.writer(f)
  for i in range(len(data[0][0])):
    dataWriter.writerow([data[0][0][i],data[0][1][i],result_data[0][1][i]])
  f.close()

def outputsummary(datafile,resultfile,outputfilename):
  data=load_data_and_labels(datafile)
  result_data=load_data_and_labels(resultfile)
  
  f = open(f'{outputfilename}', 'w', encoding='utf-8', newline='')
  dataWriter = csv.writer(f)
  for i in range(len(data[0][0])):
    dataWriter.writerow([data[0][0][i],data[0][1][i],result_data[0][1][i]])
  f.close()