# ABCI_mT5

バッチスクリプトを実行するときにターミナルで実行する
2倍早くなる??
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
もしくは
pip install -v --no-cache-dir ./
```



main.pyを参考に新しいnew_tokenizerを置いて書き換えようとしている。

# mask_T5.pyができました(未検証)
ローカル --> リモート にデータを移動させたい
```
scp /Users/t_kajiura/Git/add_training/output_13.txt abci:acd13734km
```
リモート --> ローカル
```
scp abci:result.zip .
```

## 書き換えで苦戦している箇所

"add_training/ABCI_mT5_add/mask_new.py"の内容 --> Xmasバージョンのnew_tokenizerを返してくれる関数

1. mask_new.pyをimportして、tokenizerを変更
```
import mask_new

tokenizer=MT5Tokenizer(PRETRAINED_MODEL_NAME, is_fast=True) #元々あった文
new_tokenizer = mask_new.masked(tokenizer) #定義する新しいnew_tokenizer
tokenizer.add_tokens(additional_special_tokens)
```

2. データについて
データが2列のtgt,srcで与えるのを想定しているのに対して、mask_new.pyが同じ文字列からsrcとtgtを交互に出力するものを使いたい
```
def masked(tokenizer):
  def new_tokenizer(sentence):
    return inputs
  return new_tokenizer
~mask.pyより~
src = tokenizer.decode(dic["input_ids"].squeeze().tolist())
tgt = tokenizer.decode(dic["target_ids"].squeeze().tolist())
inputs = {"input_ids":にゃにゃにゃ, "attention_mask":にゃにゃにゃ, "target_ids":にゃにゃにゃ}
```

3. new_tokenizerとMT5Tokenizerを使い分けなければいけない??
- new_tokenizerを使うところ
    
    文章をちぎってmaskのsrcとtgtのデータを作成するところ
- MT5Tokenizerを使うところ

    decode, encode, add_tokensなどTokenizerに備わっているフクザツなところ

## 実行方法
つむさん作のABCIを参考に
```
python3 main_newtoken.py [使うデータ.tsv]
```
以下の名前のディレクトリを作成
- data
- model
- log
- result

順に実行
```
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130.1
pip3 install --user --upgrade pip
pip3 install -r requirements.txt
```


# TODO
- 一行のtsv(.txtが良い)で動くように改造
- トレーニングに使うデータをテキスト形式からtsvに変更する





