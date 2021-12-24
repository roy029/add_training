
#AOJ_rowの.pyファイル達を一つのテキストファイルにするコード
#python3 yomikaki.py >> aoj_row.txt

import glob
input_files = glob.glob("/Users/t_kajiura/AOJ_raw/*.py")

with open("aoj_row.txt", "w") as my_output_file:
    for input_file in input_files:
        print(input_file)
        with open(input_file) as my_input_file:
            # for line in f:
            tunagu= []
            for i,line in enumerate(my_input_file):
                #もし改行だけであればbreakをする。処理をここに書いた方が良かった...mask.pyに書いてしまった。。今後必要であれば移動します。

                line.lstrip() #文字列の冒頭の空白文字だけを取り除く
                line.rstrip() #同じく末尾の空白文字だけを取り除く
                tunagu.append(line[:-1])
            s = '\n'.join(tunagu)
            my_output_file.write(s)
my_output_file.close()