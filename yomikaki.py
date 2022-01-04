
#AOJ_rowの.pyファイル達を一つのテキストファイルにするコード
#python3 yomikaki.py >> aoj_row.txt

import glob
from tqdm import tqdm
import csv
# input_files = glob.glob("/Users/t_kajiura/AOJ_raw/*.py")

#AOJ_rowの.pyファイル達を一つのテキストファイルにするコード
def pytotxt(input_files):
    for input_file in input_files:
        with open(f'{input_file}_pytorch_tutorial.txt', "w") as my_output_file: #毎回変更
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
    # for i in range(1, 7):#ファイル名数えるだけ #毎回変更
        # with open(f"mae_{input_file}", "w") as my_output_file: #毎回変更
        for input_file in input_files:
            with open(f"{input_file}_mae.txt", "w") as my_output_file: #毎回変更
                print(input_file)
                with open(input_file) as my_input_file:
                    sentence = []
                    for i,line in enumerate(my_input_file):
                        if line.startswith("#")==True:
                            pass
                        else:
                            if line == "\n":
                                pass
                            else:
                                sentence.append(line[:-1])

                    s = '\n'.join(sentence)
                    my_output_file.write(s)
            my_output_file.close()

def sakujyo(input_files):#先頭と末尾のダブルクォーテーションを削除する Codex
    for input_file in input_files:
        with open(f'{input_file}_out.tsv', mode='w') as my_output_file:
            with open(input_file) as my_input_file:
                # print(input_file)
                reader = csv.reader(my_input_file, delimiter='\t')
                for row in reader:
                    # print(row)
                    for line in row:
                        # line = line.split('\\').split('n').split('\n').split('\'')
                        # line = line.replace("\\","").replace("n","").replace("\n","").replace("\'","")
                        line = line.lstrip('"')
                        line = line.rstrip('"')
                        # line = line.split('\\')
                        # line = "".join(line)
                        # line = line.split('n')
                        # line = "".join(line)
                        # line = line.split('\n')
                        # line = "".join(line)
                        # line = line.split('\'')
                        # line = "".join(line)
                        # line = line.split('\n')
                        
                        my_output_file.write(line)
        my_output_file.close()
input_files_tsv = glob.glob("")

def tsvtotxt(csv_file):
    with open(csv_file, "r") as my_input_file:
        with open(f'{csv_file}.txt', "w") as my_output_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()

input_files_tsv = glob.glob("/Users/t_kajiura/Git/add_training/input_file/codex_out/*.tsv")
for csv_file in tqdm(input_files_tsv):#毎回変更
    print(csv_file)
    tsvtotxt(csv_file)


## ここ実行部分だよ---------------------------
# input_files_py = glob.glob("/Users/t_kajiura/Git/add_training/tutorials/*.py")#毎回変更
# input_files_txt = glob.glob("/Users/t_kajiura/Git/add_training/torchdata.txt")#毎回変更
# input_codes_txt = glob.glob("/Users/t_kajiura/Git/add_training/codexdata/python_train_*_out.tsv")#毎回変更
# # for input_file in tqdm(input_files_py):#毎回変更
# #     # i = i+2
# #     print(input_file)
# # pytotxt(input_files_py)
# commentout(input_files_txt)
#     # sakujyo(input_codes_txt)


#自分用手順メモ1224---------------------------
# チュートリアルからダウンロードした.ipynbを[jupyter nbconvert --to python XXX.ipynb]で.pyに変換
# .pyをpytotxt(input_files)を.txtに変換
# .txtは日本語のコメント文をシャープにしてくれるので正規表現で#右側を取り除く
#-------------------------------------------