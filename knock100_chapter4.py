# 第4章: 形態素解析
#
# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，その結果をneko.txt.mecabというファイルに保存せよ．
# このファイルを用いて，以下の問に対応するプログラムを実装せよ．
#
# なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．
#
# 31. 動詞
# 動詞の表層形をすべて抽出せよ．
#
# 32. 動詞の原形
# 動詞の原形をすべて抽出せよ．
def lesson31_32(mecab_datas):
    for mecab_data in  mecab_datas:
        if mecab_data['pos'] == '動詞':
            print('表層系={0:s}、原型={1:s}'.format(mecab_data['surface'] , mecab_data['base']))

# 33. サ変名詞
# サ変接続の名詞をすべて抽出せよ．
def lesson33(mecab_datas):
    for mecab_data in  mecab_datas:
        if mecab_data['pos'] == '名詞' and mecab_data['pos1'] == 'サ変接続':
            print('名詞={0:s}、原型={1:s}'.format(mecab_data['surface'], mecab_data['base']))
#
# 34. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．
def lesson34(mecab_datas):
    for idx, mecab_data in enumerate(mecab_datas):
        if mecab_data['surface'] == 'の' and mecab_data['pos'] == '助詞' and mecab_data['pos1'] == '連体化':
            bef = mecab_datas[idx - 1]
            aft = mecab_datas[idx + 1]
            if bef['pos'] == '名詞' and bef['pos1'] == '一般' and aft['pos'] == '名詞' and aft['pos1'] == '一般':
                print('{0:s}{1:s}{2:s}'.format(bef['surface'], mecab_data['surface'], aft['surface']))
#
# 35. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
def lesson35(mecab_datas):
    nounlist = []
    for idx, mecab_data in enumerate(mecab_datas):
        if mecab_data['pos'] == '名詞':
            nounlist.append(mecab_data['surface'])
        else:
            if len(nounlist) > 1:
                print(str(nounlist))
            if len(nounlist) > 0:
                nounlist = []

#
# 36. 単語の出現頻度
# 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．
def lesson36(mecab_datas):
    wordcnt = {}
    for idx, mecab_data in enumerate(mecab_datas):
        word = mecab_data['surface']
        if word in wordcnt.keys():
            wordcnt[word] += 1
        else:
            wordcnt[word] = 1

    sortlist = sorted(wordcnt.items(), key=lambda elem: elem[1], reverse=True)
    print(sortlist[0:19])
    return sortlist

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib # ソース内では呼ばれてないが、日本語化の為に必須。

#
# 37. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
def lesson37(sortlist):
    matplotlib.use('TkAgg')
    left = []
    height = []
    label = []
    for idx, cntinfo in enumerate(sortlist[0:19]):
        left.append(idx)
        label.append(cntinfo[0])
        height.append(cntinfo[1])

    plt.bar(left, height, tick_label=label, align="center")
    # plt.rcParams['font.family'] = 'IPAPGothic'
    plt.title("Word frequency rate")
    plt.xlabel('単語')
    plt.ylabel("frequency")
    plt.grid(True)
    plt.show()
    # plt.savefig('lesson37.png', bbox_inches='tight')

#
# 38. ヒストグラム
# 単語の出現頻度のヒストグラム（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．
def lesson38(sortlist):
    freqlist = []
    for cntinfo in sortlist:
        cnt = cntinfo[1]
        freqlist.append(cntinfo[1])
    plt.hist(freqlist, log=True, bins=40)
    plt.show()

#
# 39. Zipfの法則
# 単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
def lesson39(sortlist):
    x = []
    y = []
    for idx, cntinfo in enumerate(sortlist[0:19]):
        cnt = cntinfo[1]
        x.append(idx)
        y.append(cnt)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y)
    plt.show()
# 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
# ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
# 1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．
def main():
    # tar_file = tarfile.open('jawiki-country.json.gz', 'r:gz')
    f_mecab = open('neko.txt.mecab', 'r')
    try:
        mecab_datas = []
        while True:
            s_line = f_mecab.readline().replace('\n','')
            # print(s_line)
            if not s_line:
                break
            if s_line == 'EOS':
                continue

            mecabwk = {}
            wk1 = s_line.split('\t')
            wk2 = wk1[1]
            mecabwk['surface'] = wk1[0]
            wk3 = wk2.split(',')
            mecabwk['pos'] = wk3[0]
            mecabwk['pos1'] = wk3[1]
            mecabwk['base'] = wk3[6]
            mecab_datas.append(mecabwk)

        lesson31_32(mecab_datas)
        lesson33(mecab_datas)
        lesson34(mecab_datas)
        lesson35(mecab_datas)
        lesson36(mecab_datas)
        sortlist = lesson36(mecab_datas)
        lesson37(sortlist)
        lesson38(sortlist)
        lesson39(sortlist)
        # lesson20(json_data)
        # lesson21(json_data)
        # lesson22(json_data)
        # lesson23(json_data)
        # lesson24(json_data)
        # lesson25(json_data)
        # # lesson26(json_data, 3)
        # # lesson27(json_data)
        # # lesson28(json_data)
        # lesson29(json_data)
    finally:
        f_mecab.close()
        # tar_file.close()

main()