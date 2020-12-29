# 第10章: ベクトル空間法 (II)
# 第10章では，前章に引き続き単語ベクトルの学習に取り組む．
#
# 90. word2vecによる学習
# 81で作成したコーパスに対してword2vecを適用し，単語ベクトルを学習せよ．さらに，学習した単語ベクトルの形式を変換し，86-89のプログラムを動かせ．
# time ./word2vec -train ../enwiki-20150112-400-r10-105752-81.txt -output vectors-90.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
# time ./word2vec -train ../enwiki-20150112-400-r10-105752-81-s.txt -output vectors-90s.bin -cbow 1 -size 300 -window 5 -negative 25 -hs 0 -sample 1e-5 -threads 10 -binary 1 -iter 15
#　所要時間２時間弱
# time ./word2vec -train ../enwiki-20150112-400-r10-105752-81-s.txt -output vectors-90s1.bin -cbow 1 -size 300 -window 5 -hs 0 -sample 1e-5 -threads 10 -binary 1 -iter 15
from gensim.models import KeyedVectors
import numpy as np

FNAME_WORD2VEC = './word2vec/vectors-90s1.bin'

# https://qiita.com/kenta1984/items/93b64768494f971edf86
# gesim/models/utils_any2vec.py
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_cos_sim(model, word1, word2):
    matrix_x300 = model.vectors
    idx1 = model.vocab[word1].index
    idx2 = model.vocab[word2].index
    vec1 = matrix_x300[idx1]
    vec2 = matrix_x300[idx2]
    return cos_sim(vec1, vec2)

def lesson86_87(model):
    matrix_x300 = model.vectors
    # idx_usa = model.vocab['united_states_of_america'].index
    idx_usa2 = model.vocab['the_united_states_of_america'].index
    idx_us = model.vocab['u.s'].index
    #vec_usa = matrix_x300[idx_usa]
    vec_usa2 = matrix_x300[idx_usa2]
    vec_us = matrix_x300[idx_us]
    #print(vec_usa)
    print(vec_usa2)
    print(vec_us)
    #print('usa vs usa2={}'.format(cos_sim(vec_usa, vec_usa2)))
    #print('usa vs us={}'.format(cos_sim(vec_usa, vec_us)))
    print('usa2 vs us={}'.format(cos_sim(vec_usa2, vec_us)))

def lesson88(model):
    results = model.most_similar(positive=['england'])
    for result in results:
        print(result)

# from gensim.similarities.nmslib import NmslibIndexer

def lesson90sub(idxer, model, word_a, word_b, word_c, topcnt):
    matrix_x300 = model.vectors
    idx_a = model.vocab[word_a].index
    vec_a = matrix_x300[idx_a]
    idx_b = model.vocab[word_b].index
    vec_b = matrix_x300[idx_b]
    idx_c = model.vocab[word_c].index
    vec_c = matrix_x300[idx_c]

    # vec_calc = vec_a - vec_b + vec_c
    vec_calc = vec_a - vec_b + vec_c

    return idxer.most_similar(vec_calc, topcnt)

# def lesson89(model):
#
#     idxer = NmslibIndexer(model)
#
#     results = lesson90sub(idxer, model, 'spain', 'madrid', 'athens', 10)
#     for result in results:
#        print(result)


# def lesson90():
#     model = KeyedVectors.load_word2vec_format(FNAME_WORD2VEC, binary=True)
#
#     print(model.vectors.shape)
#     # print(model.vocab)
#     idx = model.vocab['the_united_states_of_america'].index
#     print(model.vectors[idx])
#
#     # lesson86_87(model)
#     # lesson88(model)
#     lesson89(model)

# lesson90()

# 91. アナロジーデータの準備
# 単語アナロジーの評価データをダウンロードせよ．このデータ中で": "で始まる行はセクション名を表す．
# 例えば，": capital-common-countries"という行は，"capital-common-countries"というセクションの開始を表している．
# ダウンロードした評価データの中で，"family"というセクションに含まれる評価事例を抜き出してファイルに保存せよ．

# copy by manually
FNAME_RESULT91 = 'questions-familyword.txt'
#
# 92. アナロジーデータへの適用
# 91で作成した評価データの各事例に対して，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．こ
# のプログラムを85で作成した単語ベクトル，90で作成した単語ベクトルに対して適用せよ．
# real	93m58.692s
# user	86m19.440s
# sys	2m1.899s
# real	90m11.485s
# user	86m29.606s
# sys	2m10.261s

from scipy import io
import knock100_chapter9
#
from gensim.models import word2vec
from gensim import matutils  # utility fnc for pickling, common scipy operations etc
from numpy import float32 as REAL, array
FNAME_RESULT92 = 'questions-familyword-92.txt'
FNAME_RESULT92a = 'questions-familyword-92a.txt'
FNAME_RESULT92c = 'questions-familyword-92c.txt'

# model : model of word2vec
def word2vec_analogy(model, worda, wordb, wordc, idxer=None):
    result = model.most_similar(negative=[wordb],
                                positive=[worda, wordc],
                                indexer=idxer)
    return result

def lesson92():
    print('load vector 85')
    # matrix_85 = io.loadmat(knock100_chapter9.FNAME_85_MATRIX)['matrix_x300']
    matrix_85 = knock100_chapter9.load_norm_matrix()
    word_idx_85, idx_word_85 = knock100_chapter9.getTIndexObj()
    print('load vector 90')
    model_90 = KeyedVectors.load_word2vec_format(FNAME_WORD2VEC, binary=True)
    # idxer = NmslibIndexer(model_90)
    # model_90 = word2vec.Word2Vec.load(FNAME_WORD2VEC, binary=True)

    f_questions = open(FNAME_RESULT91, 'rt')
    #f_results = open(FNAME_RESULT92, 'wt')
    f_results = open(FNAME_RESULT92c, 'wt')
    try:
        cnt = 0
        for question in f_questions:
            wk = question.rstrip()
            elems = wk.split(' ')
            # result85 = knock100_chapter9.lesson89sub(matrix_85, word_idx_85, elems[1], elems[0], elems[2])[0]
            result85 = knock100_chapter9.lesson89subNew(matrix_85, word_idx_85, idx_word_85, elems[1], elems[0], elems[2])[0]
            # result90 = lesson90sub(idxer, model_90, elems[1], elems[0], elems[2], 1)[0]
            # result90 = word2vec_analogy(model_90, elems[1], elems[0], elems[2])[0]
            appenddat = []
            appenddat.append(result85[0])
            appenddat.append(str(result85[1]))
            appenddat.append('dummy')
            appenddat.append('0')
            #appenddat.append(result90[0])
            #appenddat.append(str(result90[1]))

            f_results.write(wk + ' ' + ' '.join(appenddat) + '\n')
            cnt += 1
            if cnt % 10 == 0:
                print('{0:d} words checked'.format(cnt))

    finally:
        f_questions.close()

lesson92()

FNAME_RESULT92b = 'questions-familyword-92b.txt'

def word2vec_analogyb(model, worda, wordb, wordc):
    idxa = model.vocab[worda].index
    idxb = model.vocab[wordb].index
    idxc = model.vocab[wordc].index

    # print('ANALOGY : {0:s}, {1:s}, {2:s}'.format(worda, wordb, wordc))
    mean = []
    all_idxs = []
    all_idxs.append(idxa)
    all_idxs.append(idxb)
    all_idxs.append(idxc)
    if idxa is not None:
        veca = model.word_vec(worda, use_norm=True)  # L2-normalized
        # veca = model.vectors_norm[model.vocab[worda].index]
        # veca = model.word_vec(worda, use_norm=False)  # L2-normalized
        mean.append(1.0 * veca)
    if idxb is not None:
        vecb = model.word_vec(wordb, use_norm=True)  # L2-normalized
        # vecb = model.vectors_norm[model.vocab[wordb].index]
        # vecb = model.word_vec(wordb, use_norm=False)  # L2-normalized
        mean.append(-1.0 * vecb)
    if idxc is not None:
        vecc = model.word_vec(wordc, use_norm=True)  # L2-normalized
        # vecc = model.vectors_norm[model.vocab[wordc].index]
        # vecc = model.word_vec(wordc, use_norm=False)  # L2-normalized
        mean.append(1.0 * vecc)

    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
    # mean = veca - vecb + vecc

    dists = np.dot(model.vectors_norm, mean)
    best = matutils.argsort(dists, topn=10, reverse=True)
    result = [(model.index2word[sim], float(dists[sim])) for sim in best if sim not in all_idxs]
    return result

    # temp = model.similar_by_vector(mean, 10)

    # return model.similar_by_vector(mean, 1)
    # return model.most_similar(positive=[mean])
    # temp = idxer.most_similar(mean, 1)
    # for idx, wkvec in enumerate(temp):
    #     if wkvec[0] == worda or wkvec[0] == wordb or wkvec[0] == wordc:
    #         continue
    #     return [wkvec]
    # return []

def lesson92b():
    print('load vector 90b')
    model_90 = KeyedVectors.load_word2vec_format(FNAME_WORD2VEC, binary=True)
    print('init vector 90b')
    model_90.init_sims() # for init norm
    print('init indexer 90b')
    # idxer = NmslibIndexer(model_90)
    # model_90 = word2vec.Word2Vec.load(FNAME_WORD2VEC, binary=True)

    print('start vector 90b')
    f_questions = open(FNAME_RESULT91, 'rt')
    f_results = open(FNAME_RESULT92b, 'wt')
    try:
        cnt = 0
        for question in f_questions:
            wk = question.rstrip()
            elems = wk.split(' ')
            # result90 = word2vec_analogy(model_90, elems[1], elems[0], elems[2])[0]
            result90 = word2vec_analogyb(model_90, elems[1], elems[0], elems[2])[0]
            appenddat = []
            appenddat.append('dummy')
            appenddat.append('0')
            appenddat.append(result90[0])
            appenddat.append(str(result90[1]))

            f_results.write(wk + ' ' + ' '.join(appenddat) + '\n')
            cnt += 1
            if cnt % 10 == 0:
                print('{0:d} words checked'.format(cnt))
    finally:
        f_questions.close()

# lesson92b()

#
# 93. アナロジータスクの正解率の計算
# 92で作ったデータを用い，各モデルのアナロジータスクの正解率を求めよ．
# cnt_les : 33 / 506 words correct rate=0.065217
# cnt_w2v : 45 / 506 words correct rate=0.088933
# cnt_les : 372 / 506 words correct rate=0.735178
# real	0m51.337s
# user	0m42.132s
# sys	0m7.277s

def lesson93():

    # f_results = open(FNAME_RESULT92, 'rt')
    # f_results = open(FNAME_RESULT92a, 'rt')
    # f_results = open(FNAME_RESULT92b, 'rt')
    f_results = open(FNAME_RESULT92c, 'rt')
    try:
        cnt = 0
        cnt_w2v = 0
        cnt_les = 0
        for result in f_results:
            wk = result.rstrip()
            elems = wk.split(' ')
            ans_correct = elems[3]
            ans_lesson = elems[4]
            ans_word2vec = elems[6]

            cnt += 1
            if ans_correct == ans_word2vec:
                cnt_w2v += 1
            if ans_correct == ans_lesson:
                cnt_les += 1

        print('cnt_les : {0:d} / {1:d} words correct rate={2:f}'.format(cnt_les, cnt, (cnt_les / cnt)))
        print('cnt_w2v : {0:d} / {1:d} words correct rate={2:f}'.format(cnt_w2v, cnt, (cnt_w2v / cnt)))

    finally:
        f_results.close()

lesson93()
#
# 94. WordSimilarity-353での類似度計算
# The WordSimilarity-353 Test Collectionの評価データを入力とし，1列目と2列目の単語の類似度を計算し，各行の末尾に類似度の値を追加するプログラムを作成せよ．
# このプログラムを85で作成した単語ベクトル，90で作成した単語ベクトルに対して適用せよ．
# http://alfonseca.org/eng/research/wordsim353.html

from scipy import io
# from gensim.similarities.nmslib import NmslibIndexer
# import knock100_chapter9

FNAME_WORD_SIM353_BASE = 'ws353simrel/wordsim_similarity_goldstandard.txt'
#FNAME_WORD_SIM353_94 = 'wordsim_similarity_94.txt'
FNAME_WORD_SIM353_94 = 'wordsim_similarity_94a.txt'

def lesson94():
    # print('load vector 85')
    # matrix_85 = io.loadmat(knock100_chapter9.FNAME_85_MATRIX)['matrix_x300']
    # word_idx_85 = knock100_chapter9.getTIndexObj()
    print('load vector 90')
    model_90 = KeyedVectors.load_word2vec_format(FNAME_WORD2VEC, binary=True)

    f_data= open(FNAME_WORD_SIM353_BASE, 'rt')
    f_results = open(FNAME_WORD_SIM353_94, 'wt')
    try:
        for data in f_data:
            wk = data.rstrip()
            elems = wk.split('\t')
            if len(elems) < 2:
                continue
            word1 = elems[0].lower()
            word2 = elems[1].lower()
            # sim_85 = knock100_chapter9.get_cos_sim(matrix_85, word_idx_85, word1, word2)
            # sim_90 = get_cos_sim(model_90, word1, word2)
            sim_90 = model_90.similarity(word1, word2)
            # f_results.write('{0:s}\t{1:f}\t{2:f}\n'.format(wk, sim_85, sim_90) )
            f_results.write('{0:s}\t{1:f}\t{2:f}\n'.format(wk, 0, sim_90) )

    finally:
        f_data.close()
        f_results.close()

# lesson94()

#
# 95. WordSimilarity-353での評価
# 94で作ったデータを用い，各モデルが出力する類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
from scipy.stats import spearmanr

def lesson95():
    f_data= open(FNAME_WORD_SIM353_94, 'rt')
    try:
        ary_85 = []
        ary_90 = []
        for data in f_data:
            wk = data.rstrip()
            elems = wk.split('\t')
            if len(elems) < 2:
                continue
            val_85 = float(elems[3])
            val_90 = float(elems[4])
            ary_85.append(val_85)
            ary_90.append(val_90)
        correlation, pvalue = spearmanr(ary_85, ary_90)
        print('correlation={0:f}, pvalue={1:f} '.format(correlation, pvalue))

    finally:
        f_data.close()

# lesson95()

#
# 96. 国名に関するベクトルの抽出
# word2vecの学習結果から，国名に関するベクトルのみを抜き出せ．
import pickle

FNAME_COUNTRY_VEC = 'country_vec_96.dump'
FNAME_COUNTRY_IDX = 'country_idx_96.dump'
#
# def lesson96():
#     model_90 = KeyedVectors.load_word2vec_format(FNAME_WORD2VEC, binary=True)
#
#     f_data= open('countrynames_all.txt', 'rt')
#     try:
#         country_vec = np.empty([0, 300], dtype=np.float64)
#         country_idx = []
#         for data in f_data:
#             countlyname = data.rstrip().lower().replace(' ', '_')
#
#             if countlyname in model_90.vocab.keys() and countlyname not in country_idx:
#                 idx = model_90.vocab[countlyname].index
#                 vec = model_90.vectors[idx]
#                 country_vec = np.vstack([country_vec, vec])
#                 country_idx.append(countlyname)
#
#         with open(FNAME_COUNTRY_VEC, 'wb') as f_vec:
#             pickle.dump(country_vec, f_vec)
#         with open(FNAME_COUNTRY_IDX, 'wb') as f_idx:
#             pickle.dump(country_idx, f_idx)
#
#     finally:
#         f_data.close()

# lesson96()

#
# 97. k-meansクラスタリング
# 96の単語ベクトルに対して，k-meansクラスタリングをクラスタ数k=5として実行せよ．

from sklearn.cluster import KMeans

FNAME_COUNTRY_CLASS = 'country_class_97.dump'

def lesson97():
    with open(FNAME_COUNTRY_VEC, 'rb') as data_file:
        country_vec = pickle.load(data_file)

    pred = KMeans(n_clusters=5).fit_predict(country_vec)
    with open(FNAME_COUNTRY_CLASS, 'wb') as f_class:
        pickle.dump(pred, f_class)

    print(pred)

# lesson97()

#
# 98. Ward法によるクラスタリング
# 96の単語ベクトルに対して，Ward法による階層型クラスタリングを実行せよ．さらに，クラスタリング結果をデンドログラムとして可視化せよ．
# https://analysis-navi.com/?p=1884
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def lesson98():
    with open(FNAME_COUNTRY_VEC, 'rb') as data_file:
        country_vec = pickle.load(data_file)
    with open(FNAME_COUNTRY_IDX, 'rb') as f_country_idx:
        country_idx = pickle.load(f_country_idx)
    print(country_vec.shape)
    print(country_idx)

    linkage_result = linkage(country_vec, method='ward', metric='euclidean')

    plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
    dendrogram(linkage_result, labels=country_idx)
    plt.show()

# lesson98()
#
# 99. t-SNEによる可視化
# 96の単語ベクトルに対して，ベクトル空間をt-SNEで可視化せよ．
# https://qiita.com/stfate/items/8988d01aad9596f9d586
# http://inaz2.hatenablog.com/entry/2017/01/24/211331

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def lesson99():
    with open(FNAME_COUNTRY_VEC, 'rb') as data_file:
        country_vec = pickle.load(data_file)
    with open(FNAME_COUNTRY_IDX, 'rb') as f_country_idx:
        country_idx = pickle.load(f_country_idx)
    with open(FNAME_COUNTRY_CLASS, 'rb') as f_class:
        pred = pickle.load(f_class)
    print(pred)


    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(country_vec)

    print(X_reduced.shape)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set1')
    for idx, val in enumerate(country_idx):
        cval = cmap(pred[idx])
        ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], color=cval)
        ax.annotate(val, xy=(X_reduced[idx, 0], X_reduced[idx, 1]), color=cval)
    plt.show()

#lesson99()