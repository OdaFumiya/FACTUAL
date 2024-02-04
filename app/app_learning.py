import os
import shutil
import glob
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import makeCSVtoConll
from learning import learning
import json
import traceback
import threading


def StartLearning(): 
  global nowprocess
     
  nowprocess = True  
  global filelist
  conllfilelist=[]
  otherfilelist=[]  
  for f in filelist:
    if (f.endswith(".conll")):
      conllfilelist.append(f)
    else:otherfilelist.append(f)
  
  print(f"conll:{conllfilelist},other:{otherfilelist}")
  makeCSVtoConll.CSVFileListtoConllFileList(otherfilelist,"conllfile\\tmp")
  makeCSVtoConll.CSVFileListtoConllFileList(otherfilelist,"conllfile\\files")

  trainfilelist=conllfilelist
  trainfilelist.extend(glob.glob("conllfile\\tmp\\*"))

  with open("conllfile\\tmp\\tmp_learning_file.conll", "w",encoding="UTF-8") as outfile:
    for file_name in trainfilelist:
        print(file_name)
        with open(file_name, "r",encoding="UTF-8") as infile:
            outfile.write(infile.read())
  bat = 4
  num = 100
  interval = 1
  print(trainfilelist)
  if(True):
    try:
        train = "conllfile\\tmp\\tmp_learning_file.conll"
        name = modelnameentry.get()

        result = "result\\" + name

        print(f"学習を開始します。\n教師ファイル:{train}\n学習回数:{num}回\n記録インターバル:{interval}回\nプロセス番号:{os.getpid()}")
        
        tag_to_ix,word_to_ix =learning(train, result, num,interval,bat)
        filedata={"train":train,"tag_to_ix":tag_to_ix,"word_to_ix":word_to_ix,}
        with open(f'{result}/datafile.json', 'w',newline="",encoding="utf-8") as f:
          json.dump(filedata, f, ensure_ascii=False, indent=4)
          f.close()
        messagebox.showinfo('確認', f"\n学習および結果の出力が完了しました。\nファイルは{result}として記録されました。")

    except Exception as e:
        tk.Tk().withdraw()
        traceback.print_exc()
        messagebox.showinfo('エラー', '学習が中断されました。')
  
  shutil.rmtree("conllfile\\tmp")
  os.mkdir("conllfile\\tmp")
  learningbutton["text"]="学習開始"
  nowprocess = False

def Learningtargetbuttonclick():
  if nowprocess:
    return
  global filelist
  conllfiles=filedialog.askopenfilename(
    title = "分析対象を開く",
    initialdir = os.getcwd(), #開くディレクトリを指定
    filetypes = {("学習データファイル", "*.conll;*.csv")}, 
    multiple =True
    )
  filelisttextbox.configure(state='normal')
  filelisttextbox.delete("1.0","end")
  for t in conllfiles:
    filelisttextbox.insert(tk.END, os.path.basename(t)+"\n")
  filelisttextbox.configure(state='disable')
  filelist=conllfiles

def learningbuttonclick():
  if nowprocess:
    return
  learningbutton["text"]="学習中……"
  thread = threading.Thread(target=StartLearning)
  thread.start()



if __name__ == '__main__':
  filelist=[]
  nowprocess = False
  # Tkクラス生成
  frm = tk.Tk()
  # 画面サイズ
  frm.geometry('600x400')
  # 画面タイトル
  frm.title('学習用ツール')
  # 画面をそのまま表示

  side = tk.Frame(frm,width=200)
  Learningtargetbutton = tk.Button(side, text = '学習対象選択', font=("",18),command = Learningtargetbuttonclick, width= 16, height = 1)
  learningbutton = tk.Button(side, text = '学習開始', font=("",18),command = learningbuttonclick, width= 16, height = 1)
  Learningtargetbutton.pack(anchor = tk.N)
  learningbutton.pack(anchor = tk.N)
  side.pack(side = tk.LEFT, fill = tk.Y)


  mainfrm = tk.Frame(frm)
  modelnameentry = tk.Entry(mainfrm,font=("",12),width=40,)
  filelisttextbox = tk.Text(mainfrm,font=("",12),width=40,)
  modelnameentry.insert('end', "出力モデル名")
  filelisttextbox.insert('end', "学習対象のファイルを選択してください")
  modelnameentry.pack(anchor = tk.N)
  filelisttextbox.pack(anchor = tk.N)
  mainfrm.pack(expand = True)


  frm.mainloop()