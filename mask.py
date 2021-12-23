# import os
import random
import glob
import csv
import time
from tqdm import tqdm

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


ID = 250099

def masking(input_ids, masked):
  c = 0
  prev_index = None
  for index in masked:
    if prev_index == index - 1:
      input_ids[index] = None
    else:
      input_ids[index] = ID - c
      c += 1
    prev_index = index
  return tokenizer.decode([ids for ids in input_ids if ids != None])

def mask(s, ratio=0.15):
  inputs = tokenizer(s, return_tensors="pt")   # input のtensor 列
  input_ids = inputs["input_ids"].squeeze().tolist()[:-1] # </s> を除いたinputs のlist
  n_tokens = len(input_ids)   # 字句数
  n = max(int((n_tokens / 2) * ratio), 1)

  input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
  output_masked = list(set(list(range(0, n_tokens))) - set(input_masked))

  source = masking(input_ids[:], input_masked)
  target = masking(input_ids[:], output_masked)

  return source, target

#入力ファイルの種類によって読み込み方法を変える必要アリ...?かもしれない
# def read_txt(input_file, output_file='out.tsv'):
#   with open(input_file) as f:
#     with open(output_file, mode='w') as fo:
#         if f.split(".")[-1] == ("csv" or "tsv"):
#             tsv_writer = csv.writer(fo, delimiter='\t')
#             for line in f.readlines():
#                 src, tgt = mask(line.strip())
#                 tsv_writer.writerow([src, tgt])
#         elif f.split(".")[-1] == "txt":
#             for line in f.readlines():
#                 src, tgt = mask(line.strip())
#                 fo.write([src, tgt])
#         else:
#             print("ファイルの形を確認する")

def read_txt(input_file, output_file='aoj_row_out.tsv'):
    with open(input_file) as f:
        with open(output_file, mode='w') as fo:
            tsv_writer = csv.writer(fo, delimiter='\t')
            for line in f.readlines():
                src, tgt = mask(line.strip())
                tsv_writer.writerow([src, tgt])


def only_read_txt(input_file, output_file='out.tsv'):
  with open(input_file) as f:
    for line in f.readlines():
      src, tgt = mask(line.strip())
      print('org:', line.strip())
      print('src:', src)
      print('tgt:', tgt)
      print('===============================')

#入力用フォルダを作ったので、余裕があれば保存先フォルダを作りたい
input_files = glob.glob("/Users/t_kajiura/Git/add_training/input_file/aoj*.txt")
for input_file in tqdm(input_files):
    print(input_file)
    read_txt(input_file, output_file='aoj_row_out.tsv')
    # slack.notify(text="ファイルの実行が一つ終わったよ") # 指定したチャンネルに送信

