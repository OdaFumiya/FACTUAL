import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from prediction import predictionFromModel



def Setmodelbuttonclick():
  if nowprocess:
    return
  modelfile=filedialog.askopenfilename(
    title = "モデルを開く",
    initialdir = os.getcwd(), #開くディレクトリを指定
    filetypes = {("model file", ".pth"), ("PTH", ".pth")},
    multiple = False
    )
  modelnameentry.configure(state='normal')
  modelnameentry.delete(0, tk.END)
  modelnameentry.insert(tk.END, modelfile)
  modelnameentry.configure(state='readonly')
def Setixbuttonclick():
  modelfile=filedialog.askopenfilename(
    title = "モデルを開く",
    initialdir = os.getcwd(), #開くディレクトリを指定
    filetypes = {("data file", ".json")},
    multiple = False
    )
  ixnameentry.configure(state='normal')
  ixnameentry.delete(0, tk.END)
  ixnameentry.insert(tk.END, modelfile)
  ixnameentry.configure(state='readonly')


def Analyzetargetbuttonclick():
  global filelist
  if nowprocess:
    return
  requirementfiles=filedialog.askopenfilename(
    title = "分析対象を開く",
    initialdir = os.getcwd(), #開くディレクトリを指定
    filetypes = {("requirement file", ".txt .csv")}, 
    multiple =True
    )
  filelisttextbox.configure(state='normal')
  filelisttextbox.delete("1.0","end")
  for t in requirementfiles:
    filelisttextbox.insert(tk.END, os.path.basename(t)+"\n")
  filelisttextbox.configure(state='disable')
  filelist=requirementfiles

def Analyzebuttonclick():
  global filelist
  global nowprocess
  if nowprocess:
    return
  nowprocess = True  
  print(filelist)
  predictionFromModel(modelnameentry.get(),ixnameentry.get(),list(filelist))
  messagebox.showinfo('確認', '完了')
  nowprocess = False





if __name__ == '__main__':
  filelist=[]
  nowprocess = False
  # Tkクラス生成
  frm = tk.Tk()
  # 画面サイズ
  frm.geometry('600x400')
  # 画面タイトル
  frm.title('タグ付けツール')
  # 画面をそのまま表示
  side = tk.Frame(frm,width=200)
  Setmodelbutton = tk.Button(side, text = 'モデル選択', font=("",18),command = Setmodelbuttonclick, width= 16, height = 1)
  Setixbutton = tk.Button(side, text = 'データファイル選択', font=("",18),command = Setixbuttonclick, width= 16, height = 1)
  Analyzetargetbutton = tk.Button(side, text = '分析対象選択', font=("",18),command = Analyzetargetbuttonclick, width= 16, height = 1)
  Analyzebutton = tk.Button(side, text = '分析開始', font=("",18),command = Analyzebuttonclick, width= 16, height = 1)
  Setmodelbutton.pack(anchor = tk.N)
  Setixbutton.pack(anchor = tk.N)
  Analyzetargetbutton.pack(anchor = tk.N)
  Analyzebutton.pack(anchor = tk.N)
  side.pack(side = tk.LEFT, fill = tk.Y)


  mainfrm = tk.Frame(frm)
  modelnameentry = tk.Entry(mainfrm,font=("",12),width=40,)
  ixnameentry = tk.Entry(mainfrm,font=("",12),width=40,)
  filelisttextbox = tk.Text(mainfrm,font=("",12),width=40,)
  modelnameentry.insert('end', "models/model.pth")
  modelnameentry.configure(state='readonly')
  ixnameentry.insert('end', "models/filedata.json")
  ixnameentry.configure(state='readonly')
  filelisttextbox.insert('end', "分析対象のファイルを選択してください")
  modelnameentry.pack(anchor = tk.N)
  ixnameentry.pack(anchor = tk.N)
  filelisttextbox.pack(anchor = tk.N)
  mainfrm.pack(expand = True)


  frm.mainloop()