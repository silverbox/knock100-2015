# hightemp.txtは，日本の最高気温の記録を「都道府県」「地点」「℃」「日」のタブ区切り形式で格納したファイルである．
# 以下の処理を行うプログラムを作成し，hightemp.txtを入力ファイルとして実行せよ．
# さらに，同様の処理をUNIXコマンドでも実行し，プログラムの実行結果を確認せよ．
#
# 10. 行数のカウント
# 行数をカウントせよ．確認にはwcコマンドを用いよ．
def lesson10(input):
    print(len(input))
    # cat hightemp.txt | wc -l
    return

# 11. タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．
def lesson11(input):
    for line in input:
        print(line.replace('\t',' '))
    # sed 's/\t/ /g' hightemp.txt
    # cat hightemp.txt | tr '\t' ' '
    # expand -t 1 hightemp.txt
    return

#
# 12. 1列目をcol1.txtに，2列目をcol2.txtに保存
# 各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．
def lesson12(input):
    out_col1 = open("col1.txt", "w")
    out_col2 = open("col2.txt", "w")
    try:
        for line in input:
            cols = line.split('\t')
            out_col1.write(cols[0] +'\n')
            out_col2.write(cols[1] +'\n')
    finally:
        out_col1.close()
        out_col2.close()
    # cut -f 1 hightemp.txt
    # cut -f 2 hightemp.txt
    return

#
# 13. col1.txtとcol2.txtをマージ
# 12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ．
def lesson13(input):
    f_col1 = open("col1.txt", "r")
    f_col2 = open("col2.txt", "r")
    f_out = open("lessson13.txt", "w")
    try:
        f1_lines = f_col1.readlines()
        f2_lines = f_col2.readlines()
        for col1, col2 in zip(f1_lines, f2_lines):
            f_out.write(col1.replace('\n','') + '\t' + col2)
    finally:
        f_col1.close()
        f_col2.close()
        f_out.close()

    # paste col1.txt col2.txt
    return

#
# 14. 先頭からN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ．
def lesson14(input):
    return

#
# 15. 末尾のN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．
def lesson15(input):
    return

#
# 16. ファイルをN分割する
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．同様の処理をsplitコマンドで実現せよ．
import math
def lesson16(input, sp_cnt):
    size = math.ceil(len(input) / sp_cnt)
    for idx in range(sp_cnt):
        stidx = idx * size
        edidx = (idx + 1) * size - 1
        f_out = open('lessson16_{0:d}.txt'.format(idx), "w")
        try:
            f_out.writelines(input[stidx:edidx])
        finally:
            f_out.close()
    # split -l $((`cat hightemp.txt | wc -l`/3)) -d hightemp.txt lessson16_z
    return

#
# 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはsort, uniqコマンドを用いよ．
def lesson17(input):
    f_col1 = open("col1.txt", "r")
    try:
        f1_lines = f_col1.readlines()
        return set(f1_lines)
    finally:
        f_col1.close()
    # sort col1.txt | uniq
    return
#
# 18. 各行を3コラム目の数値の降順にソート
# 各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．
# 確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．
# https://eng-entrance.com/linux-command-sort
def lesson18(input):
    return sorted(input, key=lambda elem: float(elem.split('\t')[2]), reverse=True)
    # sort hightemp.txt --key=3,3 --numeric-sort --reverse

#
# 19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
# 各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．
import collections
def lesson19(input):
    f_col1 = open("col1.txt", "r")
    try:
        f1_lines = f_col1.readlines()
        return collections.Counter(f1_lines)
    finally:
        f_col1.close()
    # cut -f 1 hightemp.txt | sort | uniq
    # for tgt in `cut -f 1 hightemp.txt | sort | uniq`; grep -c $tgt hightemp.txt; done
    # ↑は失敗作。↓が正しい
    # cut --fields=1 hightemp.txt | sort | uniq --count | sort --reverse

def main():
    test_data = open("hightemp.txt", "r")
    try:
        test_lines = test_data.readlines()
        lesson10(test_lines)
        lesson11(test_lines)
        lesson12(test_lines)
        lesson13(test_lines)
        lesson14(test_lines)
        lesson15(test_lines)
        lesson16(test_lines, 3)
        print(lesson17(test_lines))
        print(lesson18(test_lines))
        print(lesson19(test_lines))
    finally:
        test_data.close()

main()