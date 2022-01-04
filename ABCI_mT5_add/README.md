# ABCI_mT5

main.pyを参考に新しいnew_tokenizerを置いて書き換えようとしている。

## 書き換えで苦戦している箇所

"add_training/ABCI_mT5_add/mask_new.py"の内容 --> Xmasバージョンのnew_tokenizerを返してくれる関数

1. mask_new.pyをimportして、tokenizerを変更
```
import mask_new

tokenizer=MT5Tokenizer(PRETRAINED_MODEL_NAME, is_fast=True) #元々あった文
new_tokenizer = mask_new.mask_new.masked(tokenizer) #定義する新しいnew_tokenizer
tokenizer.add_tokens(additional_special_tokens)
```

2. データについて
データが2列のtgt,srcで与えるのを想定しているのに対して、mask_new.pyが同じ文字列からsrcとtgtを交互に出力するものを使いたい

## 実行方法
つむさん作のABCIを参考に
```
python3 main_newtoken.py [使うデータ.txt]
```

# TODO
- 一行のtsvで動くように改造
- トレーニングに使うデータをテキスト形式からtsvに変更する





