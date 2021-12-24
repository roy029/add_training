
#AOJ_rowの.pyファイル達を一つのテキストファイルにするコード
#python3 yomikaki.py >> aoj_row.txt

import glob
from tqdm import tqdm
# input_files = glob.glob("/Users/t_kajiura/AOJ_raw/*.py")


#AOJ_rowの.pyファイル達を一つのテキストファイルにするコード
def pytotxt(input_files):
    for i in range(2, 9):#ファイル名数えるだけ
        with open(f"{i}_pytorch_tutorial.txt", "w") as my_output_file:
            for input_file in input_files:
                print(input_file)
                with open(input_file) as my_input_file:
                    # for line in f:
                    tunagu= []
                    for i,line in enumerate(my_input_file):
                        #もし改行だけであればbreakをする。処理をここに書いた方が良かった...mask.pyに書いてしまった。。今後必要であれば移動します。
                        line.rstrip('#')
                        line.lstrip() #文字列の冒頭の空白文字だけを取り除く
                        line.rstrip() #同じく末尾の空白文字だけを取り除く
                        tunagu.append(line[:-1])
                    s = '\n'.join(tunagu)
                    my_output_file.write(s)
        my_output_file.close()

def commentout(input_files):
    for i in range(2, 9):#ファイル名数えるだけ
        with open(f"mae_{i}_pytorch_tutorial.txt", "w") as my_output_file:
            for input_file in input_files:
                print(input_file)
                with open(input_file) as my_input_file:
                    sentence = []
                    for i,line in enumerate(my_input_file):
                        if "#" in line:
                            pass
                        else:
                            if line == "\n":
                                pass
                            else:
                                sentence.append(line[:-1])

                    s = '\n'.join(sentence)
                    my_output_file.write(s)
        my_output_file.close()

## ここ実行部分だよ---------------------------
input_files_py = glob.glob("/Users/t_kajiura/Git/add_training/pytorch_tutorials_row/*.py")
input_files_txt = glob.glob("/Users/t_kajiura/Git/add_training/mae_pytorch_tutorial/*_pytorch_tutorial.txt")
for input_file in tqdm(input_files_txt):#毎回変更
    # i = i+2
    print(input_file)
    # pytotxt(input_files_py)
    commentout(input_files_txt)#毎回変更

#自分用手順メモ1224---------------------------
# チュートリアルからダウンロードした.ipynbを[jupyter nbconvert --to python XXX.ipynb]で.pyに変換
# .pyをpytotxt(input_files)を.txtに変換
# .txtは日本語のコメント文をシャープにしてくれるので正規表現で#右側を取り除く
#-------------------------------------------