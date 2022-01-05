# ABCI_mT5

main.pyを参考に新しいnew_tokenizerを置いて書き換えようとしている。

# mask_T5.pyができました(未検証)


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


# TODO
- 一行のtsv(.txtが良い)で動くように改造
- トレーニングに使うデータをテキスト形式からtsvに変更する





