# 第8章: 機械学習
#
# 本章では，Bo Pang氏とLillian Lee氏が公開しているMovie Review Dataのsentence polarity dataset v1.0を用い，文を肯定的（ポジティブ）もしくは否定的（ネガティブ）に分類するタスク（極性分析）に取り組む．
# 70. データの入手・整形
#
# 文に関する極性分析の正解データを用い，以下の要領で正解データ（sentiment.txt）を作成せよ．
#
#     rt-polarity.posの各行の先頭に"+1 "という文字列を追加する（極性ラベル"+1"とスペースに続けて肯定的な文の内容が続く）
#     rt-polarity.negの各行の先頭に"-1 "という文字列を追加する（極性ラベル"-1"とスペースに続けて否定的な文の内容が続く）
#     上述1と2の内容を結合（concatenate）し，行をランダムに並び替える
#
# sentiment.txtを作成したら，正例（肯定的な文）の数と負例（否定的な文）の数を確認せよ．
#
# sed 's/^/-1 /'  rt-polaritydata/rt-polaritydata/rt-polarity.neg > rt-polarity-aft.neg
# sed 's/^/+1 /'  rt-polaritydata/rt-polaritydata/rt-polarity.pos > rt-polarity-aft.pos
# cat rt-polarity-aft.pos rt-polarity-aft.neg > rt-polarity-aft.all
# cat rt-polarity-aft.all | sort -R > sentiment.txt
# cat sentiment.txt | grep +1 | wc -l
# cat sentiment.txt | grep '\-1' | wc -l
#
# 71. ストップワード
# 英語のストップワードのリスト（ストップリスト）を適当に作成せよ．
# さらに，引数に与えられた単語（文字列）がストップリストに含まれている場合は真，それ以外は偽を返す関数を実装せよ．
# さらに，その関数に対するテストを記述せよ．
stoplist = set([])
f_stoplist = open('stoplist.txt', 'rt')
try:
    for s_line in f_stoplist:
        s_line = s_line.replace('\n', '')
        stoplist.add(s_line)
finally:
    f_stoplist.close()

# test at test_knock100_chapter8.py
def isStopWord(target):
    return target.lower() in stoplist

#
# 72. 素性抽出
# 極性分析に有用そうな素性を各自で設計し，学習データから素性を抽出せよ．
# 素性としては，レビューからストップワードを除去し，各単語をステミング処理したものが最低限のベースラインとなるであろう．
from nltk import stem
stemmer = stem.PorterStemmer()

def getFeatureWords(line):
    ret = []
    wkwords = line.split(' ')
    for orgword in wkwords[1:]:
        wkstem = stemmer.stem(orgword)
        if not isStopWord(wkstem):
            ret.append(wkstem)
    return (float(wkwords[0]) + 1) / 2, ret

def getWordFrequency(feature_words_list):
    wordcnt = {}
    for idx, feature_words in enumerate(feature_words_list):
        for word in feature_words:
            if word in wordcnt.keys():
                wordcnt[word] += 1
            else:
                wordcnt[word] = 1

    sortlist = sorted(wordcnt.items(), key=lambda elem: elem[1], reverse=True)
    return sortlist

import matplotlib
import matplotlib.pyplot as plt
def showWordFreq(sortlist):
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
    plt.xlabel('word')
    plt.ylabel("frequency")
    plt.grid(True)
    plt.show()

def showHistgram(sortlist):
    freqlist = []
    for cntinfo in sortlist:
        cnt = cntinfo[1]
        freqlist.append(cntinfo[1])
    plt.hist(freqlist, log=True, bins=50)
    plt.show()

def filterList(sortlist, lowestCnt):
    ret = []
    for item in sortlist:
        # if len(item[0]) < 2:
        #     continue
        if item[1] > lowestCnt:
            ret.append(item)
    print(ret[-20:])
    return ret

import numpy as np

def getOneX(words, feature_wordlist):
    feature_cnt = len(feature_wordlist)
    xi = np.zeros(feature_cnt + 1)
    # xi.append(float(1)) # for theta0
    xi[0] = float(1)  # for theta0
    for fidx, f_word in enumerate(feature_wordlist):
        if f_word[0] in words:
            # xi.append(float(1))
            xi[fidx + 1] = float(1)
        else:
            # xi.append(float(0))
            xi[fidx + 1] = float(0)
    return xi

def getMetrics(words_list, feature_wordlist):
    feature_cnt = len(feature_wordlist)
    #ret = []
    ret = np.zeros([len(words_list), feature_cnt + 1], dtype=np.float64)
    for wdidx, words in enumerate(words_list):
        #ret.append(xi)
        ret[wdidx] = getOneX(words, feature_wordlist)
    print(ret[0])
    return np.array(ret, dtype=np.float64)

#
# 73. 学習
# 72で抽出した素性を用いて，ロジスティック回帰モデルを学習せよ．
# import math
def sigmoid(z): # zはベクトル
    return 1.0 / (1.0 + np.exp(-1 * z))

def hypothesis(theta, X):
    return sigmoid(np.dot(X, theta))

def gradient(theta, X, y):
    m = len(y)
    hyp = hypothesis(theta, X)
    wkdiff = (hyp - y)    # m * 1    matrix
    # wk4sum = np.multiply(wkdiff, X)     # m * n(feature) matrix
    # print(wk4sum.shape)
    # gradwk = sum(wk4sum,1) / m # 1 * n  matrix
    # print(gradwk.T.shape)
    gradwk = np.dot(X.T, wkdiff) / m
    return gradwk.T

def costFunction(theta, X, y): # theta, yはベクトル, X は行列
    m = len(y)
    hyp = hypothesis(theta, X)

    wkz = np.log(hyp)
    wka = (-np.multiply(y, wkz))
    wkb = (1 - y)
    wkc = np.log(1 - (hyp))
    return sum(wka - np.multiply(wkb, wkc)) / m #, gradient(theta, X, y)

import scipy.optimize as op
def learnData(X, y): # theta, yはベクトル, X は行列
    m,n= X.shape
    initial_theta = np.zeros(n, dtype=np.float64)
    result = op.minimize(fun=costFunction, x0=initial_theta, args=(X, y), method='TNC', jac=gradient, options={'maxiter':1000})
    # result = op.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
    # print(result)
    # print(result[0])
    # print(result['x'].shape)
    return result


def hypothesisX(theta, data_x):
    '''仮説関数
    data_xに対して、thetaを使ってdata_yを予測

    戻り値：
    予測値の行列
    '''
    return 1.0 / (1.0 + np.exp(-data_x.dot(theta)))


def costX(theta, data_x, data_y):
    '''目的関数
    data_xに対して予測した結果と正解との差を算出

    戻り値：
    予測と正解との差
    '''
    m = data_y.size         # データ件数
    h = hypothesisX(theta, data_x)       # data_yの予測値の行列
    j = 1 / m * np.sum(-data_y * np.log(h) -
            (np.ones(m) - data_y) * np.log(np.ones(m) - h))

    return j

def gradientX(data_x, theta, data_y):
    '''最急降下における勾配の算出

    戻り値：
    thetaに対する勾配の行列
    '''
    m = data_y.size         # データ件数
    h = hypothesisX(theta, data_x)       # data_yの予測値の行列
    grad = 1 / m * (h - data_y).dot(data_x)

    return grad

def learnDataX(data_x, data_y, alpha, count):
    '''ロジスティック回帰の学習

    戻り値：
    学習済みのtheta
    '''
    theta = np.zeros(data_x.shape[1])
    c = costFunction(theta, data_x, data_y)
    grad = None
    print('\t学習開始\tcost：{}'.format(c))

    for i in range(1, count + 1):

        grad = gradientX(data_x, theta, data_y)
        theta -= alpha * grad

        # コストとthetaの最大調整量を算出して経過表示（100回に1回）
        if i % 100 == 0:
            c = costFunction(theta, data_x, data_y)
            e = np.max(np.absolute(alpha * grad))
            print('\t学習中(#{})\tcost：{}\tE:{}'.format(i, c, e))

    c = costFunction(theta, data_x, data_y)
    e = np.max(np.absolute(alpha * grad))
    print('\t学習完了 \tcost：{}\tE:{}'.format(c, e))
    print(theta)
    return theta

# 74. 予測
# 73で学習したロジスティック回帰モデルを用い，与えられた文の極性ラベル（正例なら"+1"，負例なら"-1"）と，その予測確率を計算するプログラムを実装せよ．
def reviewSentence(theta, sentence, feature_wordlist):
    y, words = getFeatureWords(sentence)
    xi = getOneX(words, feature_wordlist)
    hyp = hypothesis(theta, xi)
    print('predict:' + sentence)
    print(hyp)

#
# 75. 素性の重み
# 73で学習したロジスティック回帰モデルの中で，重みの高い素性トップ10と，重みの低い素性トップ10を確認せよ．
def topAndWorst10(theta, feature_wordlist):
    sortwklist = []
    for idx, coefficient in enumerate(theta[1:]):
        sortwk = {}
        sortwk[idx] = coefficient
        sortwklist.append(sortwk)

    sortedlist = sorted(sortwklist, key=lambda elem: list(elem.values())[0], reverse=True)
    for idx, item in enumerate(sortedlist[0:9]):
        #key = item[0] #list(item.keys())[0]
        #val = item[1] #list(item.values())[0]
        key = list(item.keys())[0]
        val = list(item.values())[0]
        print('rank {0} = {1}, value = {2}'.format(idx, feature_wordlist[key], val))

    for idx, item in enumerate(sortedlist[-9:]):
        #key = item[0] #list(item.keys())[0]
        #val = item[1] #list(item.values())[0]
        key = list(item.keys())[0]
        val = list(item.values())[0]
        print('rank {0} = {1}, value = {2}'.format(len(sortedlist) - 10 + idx, feature_wordlist[key], val))

#
# 76. ラベル付け
# 学習データに対してロジスティック回帰モデルを適用し，正解のラベル，予測されたラベル，予測確率をタブ区切り形式で出力せよ．
def labeling(theta, X, y):
    hyp = hypothesis(theta, X)
    for yi, hvali in zip(y, hyp):
        hlabel = 0
        if hvali >= 0.5:
            hlabel = 1
        # print('y={0:f}, h={1:f}, rate={2:f}'.format(yi, hlabel, hvali))

#
# 77. 正解率の計測
# 76の出力を受け取り，予測の正解率，正例に関する適合率，再現率，F1スコアを求めるプログラムを作成せよ．
def calcScoreSub(hyp, y, threthold):

    alty = 1.0 - y
    alth = 1.0 - hyp

    # tpwk = np.multiply(hyp, y)
    # tpcnt = np.count_nonzero(tpwk > threthold)
    #
    # tnwk = np.multiply(alty, alth)
    # tncnt = np.count_nonzero(tnwk > threthold)
    #
    # fpwk = np.multiply(alty, hyp)
    # fpcnt = np.count_nonzero(fpwk > threthold)
    #
    # fnwk = np.multiply(y, alth)
    # fncnt = np.count_nonzero(fnwk > threthold)

    tpcnt = 0
    tncnt = 0
    fpcnt = 0
    fncnt = 0
    for idx in range(len(y)):
        yi = y[idx]
        hi = 0
        if hyp[idx] > threthold:
            hi = 1

        if hi == 0:
            if yi == 0:
                tncnt += 1
            else:
                fncnt += 1
        else:
            if yi == 0:
                fpcnt += 1
            else:
                tpcnt += 1

    precision = tpcnt / (tpcnt + fpcnt)

    recall = tpcnt / (tpcnt + fncnt)

    f1score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1score

def calcScore(theta, X, y):
    hyp = hypothesis(theta, X)
    print(theta)
    print(hyp)
    ttlcnt = len(y)
    abs = np.abs(hyp - y)
    okcnt = np.count_nonzero(abs < 0.5)

    precision, recall, f1score = calcScoreSub(hyp, y, 0.5)

    print('ttl={0}, ok={1}, rate={2}'.format(ttlcnt, okcnt, okcnt * 100 / ttlcnt))
    print('precision={0}, recall={1}, f1score={2}'.format(precision, recall, f1score))


#
# 78. 5分割交差検定
# 76-77の実験では，学習に用いた事例を評価にも用いたため，正当な評価とは言えない．
# すなわち，分類器が訓練事例を丸暗記する際の性能を評価しており，モデルの汎化性能を測定していない．
# そこで，5分割交差検定により，極性分類の正解率，適合率，再現率，F1スコアを求めよ．
def crossValidation(X, y, count):
    m = len(X)
    step = m / count
    for idx in range(1, count):
        idxsta = int((idx - 1) * step)
        idxend = int((idx * step) - 1)
        X_eva = X[idxsta : idxend]
        y_eva = y[idxsta : idxend]
        X_trawk1 = X[0 : idxsta - 1]
        X_trawk2 = X[idxend :]
        X_tra = []
        y_trawk1 = y[0 : idxsta - 1]
        y_trawk2 = y[idxend :]
        y_tra = []
        if idx > 1:
            X_tra.extend(X_trawk1)
            y_tra.extend(y_trawk1)
        if idx < count:
            X_tra.extend(X_trawk2)
            y_tra.extend(y_trawk2)

        print(np.array(X_tra).shape)
        print(np.array(y_tra).shape)
        model = learnData(np.array(X_tra, dtype=np.float64), np.array(y_tra, dtype=np.float64))
        theta = model['x']

        print('crossvalid step {}'.format(idx))
        calcScore(theta, np.array(X_eva, dtype=np.float64), np.array(y_eva, dtype=np.float64))


#
# 79. 適合率-再現率グラフの描画
# ロジスティック回帰モデルの分類の閾値を変化させることで，適合率-再現率グラフを描画せよ．
def drawThretholdCurve(theta, X, y):
    print('start rdrawThretholdCurve 1')
    hyp = hypothesis(theta, X)
    precision_list = []
    recall_list = []
    threthold_list = [th / 20 for th in range(20)]
    for wkthethold in threthold_list:
        precision, recall, f1score = calcScoreSub(hyp, y, wkthethold)
        # threthold_list.append(wkthethold)
        precision_list.append(precision)
        recall_list.append(recall)

    print('start rdrawThretholdCurve 2')
    plt.plot(threthold_list, precision_list, label='pre', color='red')
    plt.plot(threthold_list, recall_list, label='recall', color='blue')
    plt.xlabel('threthold')
    plt.ylabel('rate')
    plt.xlim(-0.1, 1.0)
    plt.ylim(0, 1)
    plt.legend(loc=3)

    plt.show()

def main():
    f_nlp = open('sentiment.txt', 'rt')
    try:
        # print(sigmoid(np.array([1, 0.5, 0, -0.5, -1])))
        all_feature_word_set = set([])
        vec_y = []
        words_list = []
        for s_line in f_nlp:
            s_line = s_line.replace('\n','')
            y, feature_words = getFeatureWords(s_line)
            vec_y.append(y)
            words_list.append(feature_words)
            for f_wd in feature_words:
                all_feature_word_set.add(f_wd)

        print('start create frequency')
        wd_freq_list = getWordFrequency(words_list)
        # showHistgram(wd_freq_list)
        filtered_list = filterList(wd_freq_list, 5)
        print(len(filtered_list))

        print('start create matrics')
        mat_x = getMetrics(words_list, filtered_list)
        print('start learn Data')
        model = learnData(mat_x, np.array(vec_y, dtype=np.float64))
        # modelx = learnDataX(mat_x, np.array(vec_y, dtype=np.float64), 6.0, 1000)
        print('start review model')
        reviewSentence(model['x'], '-1 That movie is so boring. I fed up with it.', filtered_list)
        # reviewModelX(modelx, mat_x, np.array(vec_y, dtype=np.float64))

        # topAndWorst10(model['x'], filtered_list)
        # labeling(model['x'], mat_x, np.array(vec_y, dtype=np.float64))

        # calcScore(model['x'], mat_x, np.array(vec_y, dtype=np.float64))

        crossValidation(mat_x, vec_y, 5)

        # drawThretholdCurve(model['x'], mat_x, np.array(vec_y, dtype=np.float64))
    except:
        print(s_line)
    finally:
        f_nlp.close()

main()