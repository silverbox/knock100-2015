# 第6章: 英語テキストの処理
#
# 英語のテキスト（nlp.txt）に対して，以下の処理を実行せよ．
#
# 50. 文区切り
# (. or ; or : or ? or !) → 空白文字 → 英大文字というパターンを文の区切りと見なし，入力された文書を1行1文の形式で出力せよ．
import re

lineSepPtn = r'[.;:?!] [A-Z]'
reLineSep = re.compile(lineSepPtn)

def lineToSentence(line):
    ret = []
    re_result = reLineSep.search(line)
    wkline = line
    while re_result:
        ret.append(wkline[0:re_result.start()])
        wkline = wkline[re_result.end() - 1:]
        # print(wkline)
        re_result = reLineSep.search(wkline)

    if len(wkline) > 0:
        ret.append(wkline)

    return ret
#
# 51. 単語の切り出し
# 空白を単語の区切りとみなし，50の出力を入力として受け取り，1行1単語の形式で出力せよ．ただし，文の終端では空行を出力せよ．
def sentenceToWords(sentence_list):
    ret = []
    for sentence in sentence_list:
        ret.extend(sentence.split(' '))
    return ret

# 52. ステミング
# 51の出力を入力として受け取り，Porterのステミングアルゴリズムを適用し，単語と語幹をタブ区切り形式で出力せよ．
# Pythonでは，Porterのステミングアルゴリズムの実装としてstemmingモジュールを利用するとよい．
from nltk import stem
def steming(words):
    stemmer = stem.PorterStemmer()
    for word in words:
        stmword = stemmer.stem(word)
        print('\t'.join([word, stmword]))

#
# 53. Tokenization
# Stanford Core NLPを用い，入力テキストの解析結果をXML形式で得よ．また，このXMLファイルを読み込み，入力テキストを1行1単語の形式で出力せよ．
# ./corenlp.sh -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ../nlp.txt
import xml.etree.ElementTree as ET
def getNlpObjFromXml(filePath):
    tree = ET.parse(filePath)
    root = tree.getroot()
    # for lemma in root.iter('lemma'):
    #     print(lemma.text)

    return tree

def xmlElemToMap(elem):
    ret = {}
    for child in elem:
        tag = child.tag
        val = child.text
        ret[tag] = val
    return ret

#
# 54. 品詞タグ付け
# Stanford Core NLPの解析結果XMLを読み込み，単語，レンマ，品詞をタブ区切り形式で出力せよ．
# 55. 固有表現抽出
# 入力文中の人名をすべて抜き出せ．
def outputXml(tree):
    root = tree.getroot()
    person_list = []
    for token in root.iter('token'):
        # word = None
        # lemma = None
        # pos = None
        # ner = None
        # for child in token:
            # tag = child.tag
            # if tag == 'word':
            #     word = child.text
            # elif tag == 'lemma':
            #     lemma = child.text
            # elif tag == 'POS':
            #     pos = child.text
            # elif tag == 'NER':
            #     ner = child.text
        # print('\t'.join([word, lemma, pos]))
        vals = xmlElemToMap(token)
        print('\t'.join([vals['word'], vals['lemma'], vals['POS']]) )
        if vals['NER'] == 'PERSON':
            person_list.append(vals['word'])

    print(person_list)

#
# 56. 共参照解析
# Stanford Core NLPの共参照解析の結果に基づき，文中の参照表現（mention）を代表参照表現（representative mention）に置換せよ．
# ただし，置換するときは，「代表参照表現（参照表現）」のように，元の参照表現が分かるように配慮せよ．
def replaceMention(tree):
    root = tree.getroot()
    repstaset = set([]) # for add start mark
    # repmidset = {} # for skip output
    rependset = set([]) # for add end mark and replaced
    repstrmap = {} # for keep replace str
    for coreference in root.iterfind('./document/coreference/coreference'):
        rep_text = coreference.findtext('./mention[@representative="true"]/text')
        for mention in coreference:
            if mention.get('representative', 'false') == 'false':
                vals = xmlElemToMap(mention)
                sentence_id = int(vals['sentence'])
                start = int(vals['start'])
                end = int(vals['end']) - 1
                repstaset.add('{0}-{1}'.format(sentence_id, start))
                endkey = '{0}-{1}'.format(sentence_id, end)
                rependset.add(endkey)
                repstrmap[endkey] = rep_text

    for sentence in root.iterfind('./document/sentences/sentence'):
        sentence_id = int(sentence.get('id', '-1'))
        tokenstrlist = []
        for token in sentence.iterfind('./tokens/token'):
            token_id = int(token.get('id', '-1'))
            vals = xmlElemToMap(token)
            token_key = '{0}-{1}'.format(sentence_id, token_id)
            # print(token_key + vals['word'])
            if token_key in repstaset:
                tokenstrlist.append('＜' + vals['word'])
            elif token_key in repstrmap.keys():
                tokenstrlist.append(vals['word'] + '＞ (' + repstrmap[token_key] + ')')
            else:
                tokenstrlist.append(vals['word'])
        print(' '.join(tokenstrlist))

    # rep_text = coreference.findtext('./mention[@representative="true"]/text')

#
# 57. 係り受け解析
# Stanford Core NLPの係り受け解析の結果（collapsed-dependencies）を有向グラフとして可視化せよ．
# 可視化には，係り受け木をDOT言語に変換し，Graphvizを用いるとよい．
# また，Pythonから有向グラフを直接的に可視化するには，pydotを使うとよい．
#
# lesson44 でやったのでスキップ
#
# 58. タプルの抽出
# Stanford Core NLPの係り受け解析の結果（collapsed-dependencies）に基づき，「主語 述語 目的語」の組をタブ区切り形式で出力せよ．
# ただし，主語，述語，目的語の定義は以下を参考にせよ．
#
#     述語: nsubj関係とdobj関係の子（dependant）を持つ単語
#     主語: 述語からnsubj関係にある子（dependent）
#     目的語: 述語からdobj関係にある子（dependent）
#
def analyzeTapple(tree):
    root = tree.getroot()
    for sentence in root.iterfind('./document/sentences/sentence'):
        nsubjmap = {}  # for add start mark
        dobjmap = {}  # for add start mark
        for dep in sentence.iterfind('./dependencies[@type="collapsed-dependencies"]/dep'):
            dep_type = dep.get('type')
            gov_elem = dep.find('./governor')
            gov_idx = gov_elem.get('idx')
            dep_elem = dep.find('./dependent')
            # dep_idx = dep_elem.find('idx')
            if dep_type == 'nsubj':
                nsubjmap[gov_idx] = dep_elem.text
            if dep_type == 'dobj':
                dobjmap[gov_idx] = dep_elem.text

        for token in sentence.iterfind('./tokens/token'):
            token_id = token.get('id', '-1')
            if token_id in nsubjmap.keys() and token_id in dobjmap.keys():
                vals = xmlElemToMap(token)
                predicate = vals['word']
                nsubj = nsubjmap.get(token_id)
                dobj = dobjmap.get(token_id)
                print('\t'.join([nsubj, predicate, dobj]))

#
# 59. S式の解析
# Stanford Core NLPの句構造解析の結果（S式）を読み込み，文中のすべての名詞句（NP）を表示せよ．
# 入れ子になっている名詞句もすべて表示すること．
#
# 一旦パス

def main():
    f_nlp = open('nlp.txt', 'rt')
    try:
        sentence_list = []
        word_list = []
        for s_line in f_nlp:
            s_line = s_line.replace('\n','')
            wklines = lineToSentence(s_line)
            if len(wklines) > 0:
                sentence_list.extend(wklines)
                wkwords = sentenceToWords(wklines)
                if len(wkwords) > 0:
                    word_list.extend(wkwords)
                print(wkwords)
                steming(wkwords)

        # print(sentence_list)
        tree = getNlpObjFromXml('./nlp.txt.xml')
        outputXml(tree)
        replaceMention(tree)
        analyzeTapple(tree)
    finally:
        f_nlp.close()

main()