
# 00. 文字列の逆順
# 文字列"stressed"の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．
def lesson00(input):
    return input[::-1]
    #ret = list(input)
    #for i in range(len(ret) // 2):
    #    ret[i],ret[-1 - i] = ret[-1 - i],ret[i]
    #return ''.join(ret)

# 01. 「パタトクカシーー」
# 「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．
def lesson01(input):
    return input[::2]
    #ret = []
    #ret.append(input[0])
    #ret.append(input[2])
    #ret.append(input[4])
    #ret.append(input[6])
    #return ''.join(ret)

# 02. 「パトカー」＋「タクシー」＝「パタトクカシーー」
# 「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．
def lesson02(input_a, input_b):
    ret = []
    for a, b in zip(input_a, input_b):
        ret.append(a + b)
    return ''.join(ret)

# 03. 円周率
# "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."という文を単語に分解し，
# 各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．
def lesson03(input):
    words = input.split(' ')
    ret = []
    for word in words:
        ret.append(str(len(word.replace(',','').replace('.',''))))
    return ''.join(ret)

# 04. 元素記号
# "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
# という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭に2文字を取り出し，
# 取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．
def lesson04(input, special_idx_list):
    words = input.split(' ')
    ret = {}
    for idx, word in enumerate(words):
        wd = None
        if idx + 1 in special_idx_list:
            wd = word[0:1]
        else:
            wd = word[0:2]
        ret[wd] = idx
    return ret

# 05. n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
# この関数を用い，"I am an NLPer"という文から単語bi-gram，文字bi-gramを得よ．
def lesson05(inputlist):
    ret = []
    for idx in range(len(inputlist) - 1):
        ret.append(inputlist[idx] + inputlist[idx + 1])
    return ret

# 06. 集合
# "paraparaparadise"と"paragraph"に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
# さらに，'se'というbi-gramがXおよびYに含まれるかどうかを調べよ．
def lesson06(input_a, input_b, chkbigram):
    bigram_a = set(lesson05(input_a))
    bigram_b = set(lesson05(input_b))
    print(chkbigram in bigram_a)
    print(chkbigram in bigram_b)
    return bigram_a | bigram_b, bigram_a & bigram_b, bigram_a - bigram_b

# 07. テンプレートによる文生成
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y="気温", z=22.4として，実行結果を確認せよ．
def lesson07(x, y, z):
    return '{0:d}時の{1}は{2:.1f}'.format(x, y, z)

# 08. 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
#     英小文字ならば(219 - 文字コード)の文字に置換
#     その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．
def lesson08(input):
    charlist = list(input)
    for idx in range(len(input)):
        if charlist[idx].islower():
            byte = charlist[idx].encode()
            charlist[idx] = chr(219 - ord(charlist[idx]))
    return ''.join(charlist)

import random
# 09. Typoglycemia
# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文（例えば"I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."）を与え，その実行結果を確認せよ．
def lesson09(input):
    wordlist = input.split(' ')
    for idx, word in enumerate(wordlist):
        if len(word) > 4:
            part = list(word[1:-1])
            random.shuffle(part)
            wordlist[idx] = word[0:1] + ''.join(part) + word[-1]
    return ' '.join(wordlist)

def main():
    print(lesson00('stressed'))
    print(lesson01('パタトクカシーー'))
    print(lesson02('パトカー', 'タクシー'))
    print(lesson03('Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'))
    print(lesson04(
        'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.',
        [1, 5, 6, 7, 8, 9, 15, 16, 19]))
    print(lesson05(list('I am an NLPer')))
    print(lesson05('I am an NLPer'.split(' ')))
    add06, mul06, sub06 = lesson06('paraparaparadise', 'paragraph', 'se')
    print(add06, mul06, sub06)
    print(lesson07(12, '気温', 22.4))
    enc08 = lesson08('I couldn''t believe that I could actually understand what I was reading : the phenomenal power of the human mind .')
    print(enc08)
    print(lesson08(enc08))
    print(lesson09('I couldn''t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'))

main()