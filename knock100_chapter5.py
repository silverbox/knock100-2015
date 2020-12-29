# 第5章: 係り受け解析
#
# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をCaboChaを使って係り受け解析し，その結果をneko.txt.cabochaというファイルに保存せよ．
# このファイルを用いて，以下の問に対応するプログラムを実装せよ．
#
# 40. 係り受け解析結果の読み込み（形態素）
# 形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
# さらに，CaboChaの解析結果（neko.txt.cabocha）を読み込み，各文をMorphオブジェクトのリストとして表現し，3文目の形態素列を表示せよ．
import re

class Morph():
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    def __repr__(self):
        return '<Morph:{0}, {1}, {2}, {3}>'.format(self.surface, self.base, self.pos, self.pos1)

def line_to_morph(s_line):
    wk1 = s_line.split('\t')
    wk2 = wk1[1]
    wk3 = wk2.split(',')
    # mecabwk = {}
    # mecabwk['surface'] = wk1[0]
    # mecabwk['pos'] = wk3[0]
    # mecabwk['pos1'] = wk3[1]
    # mecabwk['base'] = wk3[6]

    return Morph(wk1[0], wk3[6], wk3[0], wk3[1])

#
# 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．
class Chunk():
    def __init__(self, line):
        self.morphs = []
        wkelems = line.split(' ')
        self.idx = int(wkelems[1])
        self.dst = int(wkelems[2].rstrip('D'))
        self.srcs = []
    def __repr__(self):
        return '<Chunk:{0}, {1}, {2}, {3}>'.format(self.morphs, self.idx, self.dst, self.srcs)

    def appendMorph(self, morph):
        self.morphs.append(morph)

    def appendSrc(self, src):
        self.srcs.append(src)

    def getTabbedWords(self):
        ret = ''
        for morph in self.morphs:
            if morph.base != 'キゴウ' and morph.pos != '補助記号':
                ret = '\t'.join([ret, morph.surface])
        return ret

    def hasNoun(self):
        for morph in self.morphs:
            if morph.pos == '名詞':
                return True, morph.pos1
        return False, None

    def hasVerb(self):
        for morph in self.morphs:
            if morph.pos == '動詞':
                return True, morph.base
        return False, None

    def getParticleList(self):
        ret = []
        for morph in self.morphs:
            if morph.pos == '助詞':
                ret.append(morph.base)
        return ret

    def getJoinStr(self):
        ret = []
        for morph in self.morphs:
            ret.append(morph.surface)
        return ''.join(ret)

#
# 42. 係り元と係り先の文節の表示
# 係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
def lesson42(sentence_list):
    for chunk_list in sentence_list:
        for chunk in chunk_list:
            if len(chunk.srcs) > 0:
                print('係り先：' + chunk.getTabbedWords())
                for src in chunk.srcs:
                    print('係り元：' + chunk_list[src].getTabbedWords())

#
# 43. 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．ただし，句読点などの記号は出力しないようにせよ．
def lesson43(sentence_list):
    for chunk_list in sentence_list:
        for chunk in chunk_list:
            hasVerb, verb = chunk.hasVerb()
            if hasVerb and len(chunk.srcs) > 0:
                for src in chunk.srcs:
                    if chunk_list[src].hasNoun():
                        print('係り先：' + chunk.getTabbedWords())
                        print('係り元：' + chunk_list[src].getTabbedWords())
#
# 44. 係り受け木の可視化
# 与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，係り受け木をDOT言語に変換し，Graphvizを用いるとよい．
# また，Pythonから有向グラフを直接的に可視化するには，pydotを使うとよい．
from graphviz import Graph
from graphviz import Digraph

def lesson44(chunk_list):
    g = Graph(format='png')
    dg = Digraph(format='png')
    for c_idx, chunk in enumerate(chunk_list):
        for m_idx, morph in enumerate(chunk.morphs):
            myid = '{0:d}-{1:d}'.format(c_idx, m_idx)
            g.node(myid, label=morph.surface)
            if m_idx != 0:
                pid = '{0:d}-0'.format(c_idx)
                g.edge(pid, myid)

    for c_idx, chunk in enumerate(chunk_list):
        if chunk.dst >= 0:
            myid = '{0:d}-0'.format(c_idx)
            pid = '{0:d}-0'.format(chunk.dst)
            g.edge(pid, myid)
    g.view()


# 45. 動詞の格パターンの抽出
# 今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
# 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． ただし，出力は以下の仕様を満たすようにせよ．
#
#     動詞を含む文節において，最左の動詞の基本形を述語とする
#     述語に係る助詞を格とする
#     述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
#
# 「吾輩はここで始めて人間というものを見た」という例文（neko.txt.cabochaの8文目）を考える．
# この文は「始める」と「見る」の２つの動詞を含み，「始める」に係る文節は「ここで」，
# 「見る」に係る文節は「吾輩は」と「ものを」と解析された場合は，次のような出力になるはずである．
#
# 始める  で
# 見る    は を
#
# このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．
#
#     コーパス中で頻出する述語と格パターンの組み合わせ
#     「する」「見る」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）

# 46. 動詞の格フレーム情報の抽出
# 45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．45の仕様に加えて，以下の仕様を満たすようにせよ．
#
#     項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
#     述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
#
# 「吾輩はここで始めて人間というものを見た」という例文（neko.txt.cabochaの8文目）を考える．
# この文は「始める」と「見る」の２つの動詞を含み，「始める」に係る文節は「ここで」，「見る」に係る文節は「吾輩は」と「ものを」と解析された場合は，次のような出力になるはずである．
#
# 始める  で      ここで
# 見る    は を   吾輩は ものを
def lesson45_46(chunk_list):
    for chunk in chunk_list:
        hasVerb, verb = chunk.hasVerb()
        if hasVerb and len(chunk.srcs) > 0:
            output_list = []
            chunkstr_list = []
            output_list.append(verb)
            for idx in chunk.srcs:
                srcchunk = chunk_list[idx]
                output_list.extend(srcchunk.getParticleList())
                chunkstr_list.append(srcchunk.getJoinStr())
            output_list.extend(chunkstr_list)
            print('\t'.join(output_list))

#
# 47. 機能動詞構文のマイニング
#
# 動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．46のプログラムを以下の仕様を満たすように改変せよ．
#
#     「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
#     述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
#     述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
#     述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
#
# 例えば「別段くるにも及ばんさと、主人は手紙に返事をする。」という文から，以下の出力が得られるはずである．
#
# 返事をする      と に は        及ばんさと 手紙に 主人は
#
# このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．
#
#     コーパス中で頻出する述語（サ変接続名詞+を+動詞）
#     コーパス中で頻出する述語と助詞パターン

def lesson47(chunk_list):
    return


#
# 48. 名詞から根へのパスの抽出
#
# 文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． ただし，構文木上のパスは以下の仕様を満たすものとする．
#
#     各文節は（表層形の）形態素列で表現する
#     パスの開始文節から終了文節に至るまで，各文節の表現を"->"で連結する
#
# 「吾輩はここで始めて人間というものを見た」という文（neko.txt.cabochaの8文目）から，次のような出力が得られるはずである．
#
# 吾輩は -> 見た
# ここで -> 始めて -> 人間という -> ものを -> 見た
# 人間という -> ものを -> 見た
# ものを -> 見た
#
# 49. 名詞間の係り受けパスの抽出
#
# 文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．
#
#     問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を"->"で連結して表現する
#     文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
#
# また，係り受けパスの形状は，以下の2通りが考えられる．
#
#     文節i
#
# から構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
# 上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節k
#
#     の内容を"|"で連結して表示
#
# 例えば，「吾輩はここで始めて人間というものを見た。」という文（neko.txt.cabochaの8文目）から，次のような出力が得られるはずである．
#
# Xは | Yで -> 始めて -> 人間という -> ものを | 見た
# Xは | Yという -> ものを | 見た
# Xは | Yを | 見た
# Xで -> 始めて -> Y
# Xで -> 始めて -> 人間という -> Y
# Xという -> Y
def main():
    f_ginza = open('neko.txt.ginza.head.txt', 'rt')
    try:
        sentence_list = []
        chunk_list = []
        wkChunk = None
        for s_line in f_ginza:
            s_line = s_line.replace('\n','')
            # print('line=' + s_line)
            if s_line == 'EOS':
                chunk_list.append(wkChunk)
                for wkchunk in chunk_list:
                    if wkchunk.dst >= 0:
                        chunk_list[wkchunk.dst].appendSrc(wkchunk.idx)
                sentence_list.append(chunk_list)
                wkChunk = None
                chunk_list = []
                continue
            elif re.match(r'^\*[^\t]+$', s_line):
                if wkChunk:
                    chunk_list.append(wkChunk)
                wkChunk = Chunk(s_line)
                continue
            elif re.match(r'^.*[^\t].*,.*,.*,.*,.*,.*,.*,.*,.*', s_line):
                wkChunk.appendMorph(line_to_morph(s_line))

        lesson42(sentence_list[3:15])
        lesson43(sentence_list[3:15])
        lesson44(sentence_list[6])
        lesson45_46(sentence_list[7])
        # lesson47(sentence_list[7])

    finally:
        f_ginza.close()
        # tar_file.close()

main()