# 第9章: ベクトル空間法 (I)
#
# enwiki-20150112-400-r10-105752.txt.bz2は，2015年1月12日時点の英語のWikipedia記事のうち，約400語以上で構成される記事の中から，
# ランダムに1/10サンプリングした105,752記事のテキストをbzip2形式で圧縮したものである．
# このテキストをコーパスとして，単語の意味を表すベクトル（分散表現）を学習したい．
# 第9章の前半では，コーパスから作成した単語文脈共起行列に主成分分析を適用し，単語ベクトルを学習する過程を，いくつかの処理に分けて実装する．
# 第9章の後半では，学習で得られた単語ベクトル（300次元）を用い，単語の類似度計算やアナロジー（類推）を行う．
#
# なお，問題83を素直に実装すると，大量（約7GB）の主記憶が必要になる．
# メモリが不足する場合は，処理を工夫するか，1/100サンプリングのコーパスenwiki-20150112-400-r100-10576.txt.bz2を用いよ．
#
# 80. コーパスの整形
# 文を単語列に変換する最も単純な方法は，空白文字で単語に区切ることである．
# ただ，この方法では文末のピリオドや括弧などの記号が単語に含まれてしまう．
# そこで，コーパスの各行のテキストを空白文字でトークンのリストに分割した後，各トークンに以下の処理を施し，単語から記号を除去せよ．
#
#     トークンの先頭と末尾に出現する次の文字を削除: .,!?;:()[]'"
#     空文字列となったトークンは削除
#
# 以上の処理を適用した後，トークンをスペースで連結してファイルに保存せよ．
import math
import re

FNAME_80_WORD = 'enwiki-20150112-400-r10-105752-word-s.txt'
def filteredWords(s_line):
    wkwords = s_line.split(' ')
    wkcaredwords = []
    for wkword in wkwords:
        wkcaredword = re.sub('^[ \.,!\?;:()\[\]\'"]*', '', wkword)
        wkcaredword = re.sub('[ \.,!\?;:()\[\]\'"]*$', '', wkcaredword)
        wkword2 = wkcaredword.lower()
        if len(wkword2) > 0:
            wkcaredwords.append(wkword2)
    return wkcaredwords


def lesson80():
    f_wikiorg = open('enwiki-20150112-400-r10-105752.txt', 'rt')
    f_wikiword = open(FNAME_80_WORD, 'wt')
    try:
        for s_line in f_wikiorg:
            wkwords = filteredWords(s_line.strip())
            if len(wkwords) > 0:
                wkline = ' '.join(wkwords)
                f_wikiword.writelines(wkline + '\n')

    finally:
        f_wikiorg.close()
        f_wikiword.close()

# lesson80()

#
# 81. 複合語からなる国名への対処
# 英語では，複数の語の連接が意味を成すことがある．
# 例えば，アメリカ合衆国は"United States"，イギリスは"United Kingdom"と表現されるが，
# "United"や"States"，"Kingdom"という単語だけでは，指し示している概念・実体が曖昧である．
# そこで，コーパス中に含まれる複合語を認識し，複合語を1語として扱うことで，複合語の意味を推定したい．
# しかしながら，複合語を正確に認定するのは大変むずかしいので，ここでは複合語からなる国名を認定したい．
#
# インターネット上から国名リストを各自で入手し，80のコーパス中に出現する複合語の国名に関して，スペースをアンダーバーに置換せよ．
# 例えば，"United States"は"United_States"，"Isle of Man"は"Isle_of_Man"になるはずである．
TREE_END_KEY = '_end_'
def getCountryNameTree():
    retmap = {}
    f_countryNameOrg = open('countrynames_sort.txt', 'rt')
    try:
        for s_line in f_countryNameOrg:
            if len(s_line) > 0:
                wkwdlist = s_line.split(' ')
                wkwdmap = retmap
                lastidx = len(wkwdlist) - 1
                for idx, word in enumerate(wkwdlist):
                    wkwd = word.rstrip().lower()
                    if not wkwd in wkwdmap.keys():
                        wkwdmap[wkwd] = {}
                    wkwdmap = wkwdmap[wkwd]
                    if idx == lastidx:
                        wkwdmap[TREE_END_KEY] = 'dmmy'

    finally:
        f_countryNameOrg.close()
    # print(retmap['the'])

    return retmap

# getCountryNameTree()

FNAME_81_WORD = 'enwiki-20150112-400-r10-105752-81-s.txt'
def lesson81():
    countryNameTree = getCountryNameTree()
    f_wikiword = open(FNAME_80_WORD, 'rt')
    f_wikijoin = open(FNAME_81_WORD, 'wt')
    # f_wikiword = open('enwiki-20150112-400-word.txt', 'rt')
    # f_wikijoin = open('enwiki-20150112-400-81.txt', 'wt')
    try:
        wkTree = countryNameTree
        for lineorg in f_wikiword:
            wordlist = []
            poollist = []
            for wordorg in lineorg.strip().split(' '):
                word = wordorg.strip()
                wkwd = word.lower()
                if wkwd in wkTree.keys():
                    poollist.append(word)
                    wkTree = wkTree[wkwd]
                    if TREE_END_KEY in wkTree.keys():
                        wordlist.append('_'.join(poollist))
                        poollist = []
                        wkTree = countryNameTree
                else:
                    if len(poollist) > 0:
                        wordlist.extend(poollist)
                        poollist = []
                        wkTree = countryNameTree

                    wordlist.append(word)

            f_wikijoin.write(' '.join(wordlist) + '\n')

    finally:
        f_wikiword.close()
        f_wikijoin.close()

lesson81()

#
# 82. 文脈の抽出
# 81で作成したコーパス中に出現するすべての単語tに関して，単語tと文脈語cのペアをタブ区切り形式ですべて書き出せ．
# ただし，文脈語の定義は次の通りとする．
#
#     ある単語tの前後d単語を文脈語cとして抽出する（ただし，文脈語に単語tそのものは含まない）
#     単語tを選ぶ度に，文脈幅dは{1,2,3,4,5} の範囲でランダムに決める．
#
import random
MAX_REL_SPAN = 5

def appendRelWord(f_wikirel, wordorg, poollist):
    span = random.randint(1, MAX_REL_SPAN)

    for wkidx in range(MAX_REL_SPAN - span, MAX_REL_SPAN):
        if len(poollist[wkidx].strip()) < 1:
            continue
        f_wikirel.writelines(wordorg + '\t' + poollist[wkidx] + '\n')
    for wkidx in range(MAX_REL_SPAN + 1, MAX_REL_SPAN + span + 1):
        if len(poollist[wkidx].strip()) < 1:
            continue
        f_wikirel.writelines(wordorg + '\t' + poollist[wkidx] + '\n')

FNAME_82_PAIRS = 'enwiki-20150112-400-r10-105752-82.txt'
def lesson82():
    f_wikijoin = open(FNAME_81_WORD, 'rt')
    f_wikirel = open(FNAME_82_PAIRS, 'wt')
    # f_wikijoin = open('enwiki-20150112-400-81.txt', 'rt')
    # f_wikirel = open('enwiki-20150112-400-82.txt', 'wt')
    try:
        poollist = []
        for idx in range(1, MAX_REL_SPAN + 1):
            poollist.append('')
        print(poollist)

        r_wdcnt = 0
        for wordorg in f_wikijoin:
            word = wordorg.strip().lower()
            if len(word) < 1:
                continue

            # pool words.
            poollist.append(word)
            if r_wdcnt < MAX_REL_SPAN:
                r_wdcnt += 1
                continue

            appendRelWord(f_wikirel, poollist[MAX_REL_SPAN], poollist)

            poollist.pop(0)
            r_wdcnt += 1

        print(poollist)
        for idx in range(1, MAX_REL_SPAN + 1):
            poollist.append('')
            appendRelWord(f_wikirel, poollist[MAX_REL_SPAN], poollist)
            poollist.pop(0)

    finally:
        f_wikijoin.close()
        f_wikirel.close()

#lesson82()

# 83. 単語／文脈の頻度の計測
# 82の出力を利用し，以下の出現分布，および定数を求めよ．
#
# f(t,c) : 単語tと文脈語c の共起回数
# f(t,∗) : 単語t の出現回数
# f(∗,c) : 文脈語c の出現回数
# N      : 単語と文脈語のペアの総出現回数
#
# sort -f enwiki-20150112-400-r10-105752-82.txt > enwiki-20150112-400-r10-105752-82-st.txt
# sort -f -k 2 enwiki-20150112-400-r10-105752-82.txt > enwiki-20150112-400-r10-105752-82-sc.txt
# N=721517276(old)
# N=674574652
def lesson83():
    re_check = re.compile(r'[A-Za-z]')

    N = 0

    wkPairCnt = 0
    wkWordCnt = 0
    wkPairWd = ''
    wkWordWd = ''
    f_wikirel = open('enwiki-20150112-400-r10-105752-82-st.txt', 'rt')
    f_wikifreq_tc = open('enwiki-20150112-400-r10-105752-83-tc.txt', 'wt')
    f_wikifreq_t = open('enwiki-20150112-400-r10-105752-83-t.txt', 'wt')
    try:
        for wordorg in f_wikirel:
            pair = wordorg.strip().lower()
            elems = pair.split('\t')
            if not re_check.match(elems[0]) or not re_check.match(elems[1]):
                continue

            if pair == wkPairWd:
                wkPairCnt += 1
            else:
                if pair < wkPairWd:
                    print('Warnpair : [' + pair + '] < [' + wkPairWd + ']')
                if wkPairCnt > 0:
                    f_wikifreq_tc.writelines('{0:s}\t{1:d}'.format(wkPairWd, wkPairCnt) + '\n')
                wkPairWd = pair
                wkPairCnt = 1

            word = elems[0]
            if word == wkWordWd:
                wkWordCnt += 1
            else:
                if word < wkWordWd:
                    print('Warnword : [' + word + '] < [' + wkWordWd + ']')
                if wkWordCnt > 0:
                    f_wikifreq_t.writelines('{0:s}\t{1:d}'.format(wkWordWd, wkWordCnt) + '\n')
                wkWordWd = word
                wkWordCnt = 1
            N += 1

        if wkPairCnt > 0:
            f_wikifreq_tc.writelines('{0:s}\t{1:d}'.format(wkPairWd, wkPairCnt) + '\n')
        if wkWordCnt > 0:
            f_wikifreq_t.writelines('{0:s}\t{1:d}'.format(wkWordWd, wkWordCnt) + '\n')

    finally:
        f_wikirel.close()
        f_wikifreq_tc.close()
        f_wikifreq_t.close()
    print('N={0:d}'.format(N))

    wkRelCnt = 0
    wkRelWd = ''
    f_wikirel_c = open('enwiki-20150112-400-r10-105752-82-sc.txt', 'rt')
    f_wikifreq_c = open('enwiki-20150112-400-r10-105752-83-c.txt', 'wt')
    try:
        for wordorg in f_wikirel_c:
            pair = wordorg.strip().lower()
            elems = pair.split('\t')
            if not re_check.match(elems[0]) or not re_check.match(elems[1]):
                continue

            word = elems[1]
            if not re_check.search(pair):
                continue
            if word == wkRelWd:
                wkRelCnt += 1
            else:
                if wkRelCnt > 0:
                    f_wikifreq_c.writelines('{0:s}\t{1:d}'.format(wkRelWd, wkRelCnt) + '\n')
                wkRelWd = word
                wkRelCnt = 1

        if wkRelCnt > 0:
            f_wikifreq_c.writelines('{0:s}\t{1:d}'.format(wkRelWd, wkRelCnt) + '\n')

    finally:
        f_wikifreq_c.close()
        f_wikirel_c.close()

#lesson83()

#
# 84. 単語文脈行列の作成
# 83の出力を利用し，単語文脈行列Xを作成せよ．ただし，行列Xの各要素Xtcは次のように定義する．
#
#     f(t,c)≥10 ならば，Xtc=PPMI(t,c)=max{logN×f(t,c)f(t,∗)×f(∗,c),0}
#     f(t,c)<10 ならば，Xtc=0
#
# ここで，PPMI(t,c) はPositive Pointwise Mutual Information（正の相互情報量）と呼ばれる統計量である．
# なお，行列Xの行数・列数は数百万オーダとなり，行列のすべての要素を主記憶上に載せることは無理なので注意すること．
# 幸い，行列Xのほとんどの要素は0になるので，非0の要素だけを書き出せばよい．
import subprocess
FNAME_83_TC = 'enwiki-20150112-400-r10-105752-83-tc.txt'
FNAME_83_C = 'enwiki-20150112-400-r10-105752-83-c.txt'
FNAME_83_T = 'enwiki-20150112-400-r10-105752-83-t.txt'
FNAME_84_SORTED_TC = 'enwiki-20150112-400-r10-105752-84-tc-wk.txt'
FNAME_84_SORTED_C = 'enwiki-20150112-400-r10-105752-84-c-wk.txt'
FNAME_84_SORTED_T = 'enwiki-20150112-400-r10-105752-84-t-wk.txt'
FNAME_84_FILTERED_TC = 'enwiki-20150112-400-r10-105752-84-tc-wk2.txt'
FNAME_84_FILTERED_C = 'enwiki-20150112-400-r10-105752-84-c-wk2.txt'
FNAME_84_FILTERED_T = 'enwiki-20150112-400-r10-105752-84-t-wk2.txt'
FNAME_84_OUTPUT = 'enwiki-20150112-400-r10-105752-84.txt'
FNAME_84_MATRIX = 'enwiki-20150112-400-r10-105752-84.mat'
FNAME_84_TLIST = 'enwiki-20150112-400-r10-105752-84-tlist.dump'
FNAME_84_CLIST = 'enwiki-20150112-400-r10-105752-84-clist.dump'

N_84 = 674574652


def sortBaseFile(input, output, idx):
    f_wiki_matrix = open(output, 'wt')
    try:
        commandlist = []
        commandlist.append('sort')
        commandlist.append('-n')
        commandlist.append('-k')
        commandlist.append(str(idx))
        commandlist.append(input)
        commandlist.append('--reverse')
        subprocess.check_call(commandlist, stdout=f_wiki_matrix)
    except Exception as e:
        print(e)
        print("subprocess.check_call() failed")
    finally:
        f_wiki_matrix.close()

#sortBaseFile(FNAME_83_TC, FNAME_84_SORTED_TC, 3)
#sortBaseFile(FNAME_83_T, FNAME_84_SORTED_T, 2)
#sortBaseFile(FNAME_83_C, FNAME_84_SORTED_C, 2)

def filterLowFreqWord(input, output, idx):
    f_wiki_tc_wk = open(input, 'rt')
    f_wiki_tc_flt = open(output, 'wt')
    try:
        for wordorg in f_wiki_tc_wk:
            pair = wordorg.rstrip()
            elems = pair.split('\t')
            count = int(elems[idx])
            if count < 10:
                break

            f_wiki_tc_flt.write(wordorg)

    finally:
        f_wiki_tc_wk.close()
        f_wiki_tc_flt.close()

#filterLowFreqWord(FNAME_84_SORTED_TC, FNAME_84_FILTERED_TC, 2)
#filterLowFreqWord(FNAME_84_SORTED_T, FNAME_84_FILTERED_T, 1)
#filterLowFreqWord(FNAME_84_SORTED_C, FNAME_84_FILTERED_C, 1)

def getCountMap(fname):
    retmap = {}
    retlist = []
    f_wiki_t_wk = open(fname, 'rt')
    try:
        idx = 0
        for wordorg in f_wiki_t_wk:
            pair = wordorg.rstrip()
            elems = pair.split('\t')
            info = {}
            info['cnt'] = int(elems[1])
            info['idx'] = idx
            if elems[0] not in retmap.keys():
                retmap[elems[0]] = info
                retlist.append(elems[0])
                idx += 1
        return retmap, retlist

    finally:
        f_wiki_t_wk.close()

import pickle
from scipy import sparse, io

def lesson84():
    map_t, list_t = getCountMap(FNAME_84_FILTERED_T)
    map_c, list_c = getCountMap(FNAME_84_FILTERED_C)
    with open(FNAME_84_TLIST, 'wb') as f_t_list:
        pickle.dump(list_t, f_t_list)
    with open(FNAME_84_CLIST, 'wb') as f_c_list:
        pickle.dump(list_c, f_c_list)
    f_wiki_tc = open(FNAME_84_FILTERED_TC, 'rt')
    f_wiki_matrix = open(FNAME_84_OUTPUT, 'wt')
    try:
        # 行列作成
        size_t = len(map_t.keys())
        size_c = len(map_c.keys())
        print('size = {}, {}'.format(size_t, size_c))
        matrix_x = sparse.lil_matrix((size_t, size_c))

        for datawk in f_wiki_tc:
            data = datawk.rstrip()
            elems = data.split('\t')
            t = elems[0]
            c = elems[1]
            tc_cnt = int(elems[2])
            t_inf = map_t[t]
            t_cnt = t_inf['cnt']
            c_inf = map_c[c]
            c_cnt = c_inf['cnt']
            calcwk = (N_84 * tc_cnt) / (t_cnt * c_cnt)
            if calcwk >= 1: # log(calcwk) >= 0
                ppmi = math.log(calcwk)
                outstr = '{0:d}\t{1:d}\t{2:f}\n'.format(t_inf['idx'], c_inf['idx'], ppmi)
                f_wiki_matrix.write(outstr)
                matrix_x[t_inf['idx'], c_inf['idx']] = ppmi

        io.savemat(FNAME_84_MATRIX, {'matrix_x': matrix_x})

    finally:
        f_wiki_tc.close()
        f_wiki_matrix.close()

# lesson84()

#
# 85. 主成分分析による次元圧縮
# 84で得られた単語文脈行列に対して，主成分分析を適用し，単語の意味ベクトルを300次元に圧縮せよ．
import sklearn.decomposition

FNAME_85_MATRIX = 'enwiki-20150112-400-r10-105752-85.mat'

def lesson85():
    # 行列読み込み
    matrix_x = io.loadmat(FNAME_84_MATRIX)['matrix_x']
    # 読込を確認
    print('matrix_x Shape:', matrix_x.shape)
    print('matrix_x Number of non-zero entries:', matrix_x.nnz)
    print('matrix_x Format:', matrix_x.getformat())

    # 次元圧縮
    clf = sklearn.decomposition.TruncatedSVD(300)
    matrix_x300 = clf.fit_transform(matrix_x)
    io.savemat(FNAME_85_MATRIX, {'matrix_x300': matrix_x300})

# メモリ4GB、swapメモリを5GBぐらい必要。実行時間は20分弱
# (ginza) ubuntu@ubuntu:~/ginza$ python3 knock100_chapter9.py
# matrix_x Shape: (680584, 597678)
# matrix_x Number of non-zero entries: 3436452
# matrix_x Format: csc
# lesson85()

# 86. 単語ベクトルの表示
# 85で得た単語の意味ベクトルを読み込み，"United States"のベクトルを表示せよ．ただし，"United States"は内部的には"United_States"と表現されていることに注意せよ．
#
# 87. 単語の類似度
# 85で得た単語の意味ベクトルを読み込み，"United States"と"U.S."のコサイン類似度を計算せよ．ただし，"U.S."は内部的に"U.S"と表現されていることに注意せよ．

def getTIndexObj():
    with open(FNAME_84_TLIST, 'rb') as data_file:
        list_t = pickle.load(data_file)
    ret_word_idx, ret_idx_word = {}, {}
    for idx, word in enumerate(list_t):
        ret_word_idx[word] = idx
        ret_idx_word[idx] = word
    return ret_word_idx, ret_idx_word

import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def lesson86_87(matrix_x300, word_idx):
    idx_usa = word_idx['united_states_of_america']
    idx_usa2 = word_idx['the_united_states_of_america']
    idx_us = word_idx['u.s']
    vec_usa = matrix_x300[idx_usa]
    vec_usa2 = matrix_x300[idx_usa2]
    vec_us = matrix_x300[idx_us]
    print(vec_usa)
    print(vec_usa2)
    print(vec_us)
    print('usa vs usa2={}'.format(cos_sim(vec_usa, vec_usa2)))
    print('usa vs us={}'.format(cos_sim(vec_usa, vec_us)))
    print('usa2 vs us={}'.format(cos_sim(vec_usa2, vec_us)))

def get_cos_sim(matrix_x300, word_idx, word1, word2):
    idx1 = word_idx[word1]
    idx2 = word_idx[word2]
    vec1 = matrix_x300[idx1]
    vec2 = matrix_x300[idx2]
    return cos_sim(vec1, vec2)

# 88. 類似度の高い単語10件
# 85で得た単語の意味ベクトルを読み込み，"England"とコサイン類似度が高い10語と，その類似度を出力せよ．
#
# [['scotland', 0.7966763304125266], ['wales', 0.718467014672938], ['yorkshire', 0.6577508602337031],
# ['somerset', 0.6475941762010358], ['australia', 0.6139684457185856], ['ireland', 0.6117049401406845],
# ['lancashire', 0.5988425331085634], ['derbyshire', 0.5836995096588632], ['new_zealand', 0.5821409920652971],
# ['essex', 0.5753786911264378]]
def findSimilarVec(matrix_x300, word_idx, vec_inspect):
    resultlist = []
    for word in word_idx.keys():
        wkidx = word_idx[word]
        wkvec = matrix_x300[wkidx]
        wkdata = []
        wkdata.append(word)
        wkdata.append(cos_sim(vec_inspect, wkvec))
        resultlist.append(wkdata)
    return sorted(resultlist, key=lambda x: x[1], reverse=True)

def lesson88(matrix_x300, word_idx):
    idx_england = word_idx['england']
    vec_ingland = matrix_x300[idx_england]

    sortedlist = findSimilarVec(matrix_x300, word_idx, vec_ingland)
    print(sortedlist[0:11])
#
# 89. 加法構成性によるアナロジー
# 85で得た単語の意味ベクトルを読み込み，vec("Spain") - vec("Madrid") + vec("Athens")を計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．
# [['spain', 0.9136016090019428], ['portugal', 0.8802865604755158], ['sweden', 0.8539706504225352],
# ['denmark', 0.8488735147943424], ['greece', 0.8449216569083768], ['belgium', 0.8404291273013003],
# ['norway', 0.8363626851350214], ['netherlands', 0.826501154104712], ['italy', 0.8083035146227875],
# ['finland', 0.8037861405387765], ['britain', 0.7951431450253504]]
def cos_sim_norm(v1, v2):
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    v1 = v1 / v1_len
    v2 = v2 / v2_len

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def findSimilarVecNorm(matrix_x300, word_idx, vec_inspect):
    resultlist = []
    for word in word_idx.keys():
        wkidx = word_idx[word]
        wkvec = matrix_x300[wkidx]
        wkdata = []
        wkdata.append(word)
        wkdata.append(cos_sim_norm(vec_inspect, wkvec))
        resultlist.append(wkdata)
    return sorted(resultlist, key=lambda x: x[1], reverse=True)

from gensim import matutils  # utility fnc for pickling, common scipy operations etc

def lesson89sub(matrix_x300, word_idx, word_a, word_b, word_c):
    idx_a = word_idx[word_a]
    vec_a = matrix_x300[idx_a]
    idx_b = word_idx[word_b]
    vec_b = matrix_x300[idx_b]
    idx_c = word_idx[word_c]
    vec_c = matrix_x300[idx_c]

    vec_calc = vec_a - vec_b + vec_c

    temp = findSimilarVecNorm(matrix_x300, word_idx, vec_calc)
    ret = []
    for idx, wkvec in enumerate(temp):
        if wkvec[0] == word_a or wkvec[0] == word_b or wkvec[0] == word_c:
            continue
        ret.append(wkvec)
    return ret

def findSimilarVecNormNew(matrix_x300, idx_word, vec_inspect):
    dists = np.dot(matrix_x300, vec_inspect)
    best = matutils.argsort(dists, topn=10, reverse=True)
    result = [(idx_word[sim], float(dists[sim])) for sim in best]
    return result

def lesson89subNew(matrix_x300, word_idx, idx_word, word_a, word_b, word_c):
    idx_a = word_idx[word_a]
    vec_a = matrix_x300[idx_a]
    idx_b = word_idx[word_b]
    vec_b = matrix_x300[idx_b]
    idx_c = word_idx[word_c]
    vec_c = matrix_x300[idx_c]

    va_len = np.linalg.norm(vec_a)
    vb_len = np.linalg.norm(vec_b)
    vc_len = np.linalg.norm(vec_c)
    vec_an = vec_a / va_len
    vec_bn = vec_b / vb_len
    vec_cn = vec_c / vc_len

    vec_calc = vec_an - vec_bn + vec_cn
    vcal_len = np.linalg.norm(vec_calc)
    vec_calc = vec_calc / vcal_len

    temp = findSimilarVecNormNew(matrix_x300, idx_word, vec_calc)
    ret = []
    for idx, wkvec in enumerate(temp):
        if wkvec[0] == word_a or wkvec[0] == word_b or wkvec[0] == word_c:
            continue
        ret.append(wkvec)
    return ret

from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax

def load_norm_matrix():
    matrix_x300 = io.loadmat(FNAME_85_MATRIX)['matrix_x300']
    dist = sqrt((matrix_x300 ** 2).sum(-1))[..., newaxis]
    return matrix_x300 / dist

def lesson89(matrix_x300, word_idx):

    print(lesson89sub(matrix_x300, word_idx, 'spain', 'madrid', 'athens')[0:11])
    print(lesson89sub(matrix_x300, word_idx, 'grandfather', 'grandmother', 'husband')[0:11])
    print(lesson89sub(matrix_x300, word_idx, 'he', 'she', 'king')[0:11])

def lesson86_89():
    matrix_x300 = io.loadmat(FNAME_85_MATRIX)['matrix_x300']
    print('matrix_x Shape:', matrix_x300.shape)
    word_idx = getTIndexObj()
    # lesson86_87(matrix_x300, word_idx)
    # lesson88(matrix_x300, word_idx)
    lesson89(matrix_x300, word_idx)

#  lesson86_89()

