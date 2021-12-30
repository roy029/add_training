# import os
import random
import glob
import csv
import time
from tqdm import tqdm
import re
import torch

#ABCIに入れる必要があるライブラリ
# !pip -q install transformers
# !pip -q install sentencepiece
# !pip -q install datasets
# # !pip -q install flax
# # !pip -q install tensorflow-gpu
# !pip install slackweb

# from slackweb import Slack
# slack = Slack("slackのwebhook")


# from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import MT5Tokenizer

# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

EOS = 1 #文章の終わりを示す
ID = 250099

def masking(input_ids, masked): #tokenizer(文章), [マスク箇所を示すindex番号] --> マスクがかかった文章ベクトル(最後にEOSがついている)
    c = 0
    prev_index = None
    for index in masked:
        if prev_index == index - 1: #マスクをかける位置でない、かつinput_idsから前の単語から続いているかをみている。
            input_ids[index] = None
        else: #マスクをかける位置の時
            input_ids[index] = ID - c 
            c += 1
        prev_index = index
    return [ids for ids in input_ids if ids != None] + [EOS] #1行ぶんのマスク文章が出てくる。

def masked(tokenizer, ratio=0.15, masking_source=masking, masking_target=masking):
  buffers = {} #データをやり取りするときに一時的にデータを溜めておく記憶場所
  def new_tokenizer(s, return_tensors="pt"): # 文章の入り口
    # print("入力文=", s)
    inputs = tokenizer(s, return_tensors=return_tensors)   # input のtensor 列

    if return_tensors != "pt":
      print(f'FIXME: return_tensors="{return_tensors}"')
      print(inputs)
      return inputs
  
    if s in buffers: #1行実行すれば中の文章が1行減ってゆく。
      target=buffers[s]
      inputs["input_ids"] = torch.tensor([target])
      inputs["attention_mask"] = inputs["input_ids"].new_ones(1, len(target)) #1だけのテンソルを作成する, shapeが1行targer列
      del buffers[s] #del文を使って不要になった変数やオブジェクトなどのデータを削除
      return inputs
    input_ids = inputs["input_ids"].squeeze().tolist()#トークナイザーをかけた文章
    
        
    if tokenizer.decode(inputs["input_ids"].squeeze()) == ("</s>" or "#</s>"):
    #   print("空白行です", inputs)
      return inputs

    if len(input_ids) == 1:  #もし改行だけだったら
      return inputs
    
    input_ids = input_ids[:-1] # </s> を除いたinputs のlist
    n_tokens = len(input_ids)   # 字句数
    n = max(int((n_tokens / 2) * ratio), 1) #マスク数
    input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
    output_masked = list(set(list(range(0, n_tokens))) - set(input_masked)) #差分を出している。[0, 1, 4, 5, 7, 10]みたいな
    source = masking_source(input_ids[:], input_masked)   #ここでmasking
    if n_tokens != 0: #文章が空白・短くてmaskトークンがない時、tgtは生成されない
      target = masking_target(input_ids[:], output_masked)   #ここでmasking
      inputs["target_ids"] = torch.tensor([target])
    # print('source', source, tokenizer.decode(source))
    # print('target', target, tokenizer.decode(target))

    buffers[s] = target #バッファに中身を詰め込む コピーなんちゃらの代わりかもしれない
    inputs["input_ids"] = torch.tensor([source])
    inputs["attention_mask"] = inputs["input_ids"].new_ones(1, len(source))
    return inputs
  
  return new_tokenizer


def read_txt(input_file, output_file='out_.tsv'):
  new_tokenizer = masked(tokenizer)
  with open(input_file) as f:
    with open(output_file, mode='w') as fo:
      tsv_writer = csv.writer(fo, delimiter='\t') #区切りでないタブ箇所で区切られてしまってペアとして取り出せない。
      for line in f.readlines():
        dic = new_tokenizer(line.strip())
        src = tokenizer.decode(dic["input_ids"].squeeze().tolist())
        if len(dic) != 1: #targetがなくペアにならない文章は書き込みをしない
          tgt = tokenizer.decode(dic["target_ids"].squeeze().tolist())
          tsv_writer.writerow([src, tgt])

def only_read_txt(input_file, output_file='out.tsv'):
    new_tokenizer = masked(tokenizer)
    with open(input_file) as f:
        for line in f.readlines():
            line = line.expandtabs(4) #タブ文字を半角空白に置換
            dic = new_tokenizer(line.strip())
            # tensor_attention = dic["attention_mask"].squeeze().tolist()
            src = tokenizer.decode(dic["input_ids"].squeeze().tolist())
            tgt = tokenizer.decode(dic["target_ids"].squeeze().tolist())

            # src, tgt = masked(line.strip())
            print('org:', line.strip())
            print('src:', src)
            print('tgt:', tgt)
            print('===============================')


#入力用フォルダを作ったので、余裕があれば保存先フォルダを作りたい
input_files = glob.glob("/Users/t_kajiura/Git/add_training/input_file/*.txt")
for input_file in tqdm(input_files):
    name = re.split('/|.txt', input_file)[-1]
    print(name)
    read_txt(input_file, output_file=f'/Users/t_kajiura/Git/add_training/output_file_noeos/{name}_out.tsv')
    # slack.notify(text="ファイルの実行が一つ終わったよ") # 指定したチャンネルに送信