# ABCI_mT5

main.pyを参考に新しいnew_tokenizerを置いて書き換えようとしている。

<書き換えで苦戦しているところ>
"add_training/ABCI_mT5_add/mask_new.py"の内容
Xmasバージョンのnew_tokenizerを返してくれる関数

mask_new.pyをimportして、tokenizer定義した後に付け足す
```
import mask_new

tokenizer=MT5Tokenizer(PRETRAINED_MODEL_NAME, is_fast=True) #元々あった文
new_tokenizer = mask_new.mask_new.masked(tokenizer) #定義する新しいnew_tokenizer
tokenizer.add_tokens(additional_special_tokens)
```


