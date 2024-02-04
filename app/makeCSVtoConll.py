import MeCab
import csv
import sys
import os
sys.path.append("..")



def csvtoconll(csvfilename):
  requirelist=[]
  conlllist=[]
  csv_file = open(f'{csvfilename}', "r", encoding="UTF-8", errors="", newline="" )
  f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
  requirelist.extend(list(f))

  tagger = MeCab.Tagger()  # 「tagger = MeCab.Tagger('-d ' + unidic.DICDIR)」

  for require in requirelist:
    if(require[1]=="<DIV>" or require[1]=="<START>"or require[1]=="<STOP>"):
      conlllist.append(f'{require[0]}\t{require[1]}\n')
      continue
    split_txt=require[0].split('\n\n')
    for i in range(len(split_txt)):
      p = tagger.parse(split_txt[i])
      result = p.split('\n')
      for j in range(len(result)):
        word=(result[j].split('\t'))[0]
        if(word=="EOS"):
          first=True
          break
        word_info=((result[j].split('\t'))[1].split(','))
        #if(word_info[0]=='助詞'):
          #continue
        if(require[1]==""):
          print(j,require[0],"ERROR")
        elif(require[1]=="O"):
          conlllist.append(f'{word}\t{require[1]}\n')
        elif(j==0):
          conlllist.append(f'{word}\tI-{require[1]}\n')
        else:
          conlllist.append(f'{word}\tB-{require[1]}\n')
  return conlllist  
  #with open(f'{conllfilename}.conll', 'w',encoding="UTF-8", newline="") as f:
  #  for conll in conlllist:
  #    f.write(conll)
  #  f.write("\n")

def CSVFileListtoConllFileList(csvfilelist,outputdir):
  filelist=[]
  for f in csvfilelist:
    filelist.append(f)
  for file in filelist:
    basename_without_ext = os.path.splitext(os.path.basename(file))[0]
    with open(f'{outputdir}/{basename_without_ext}.conll', 'w',encoding="UTF-8", newline="") as f:    
      conlllist = csvtoconll(file)
      for conll in conlllist:
        f.write(conll)
      f.write("\n")
    