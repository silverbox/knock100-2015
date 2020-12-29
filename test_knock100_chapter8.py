import unittest
import knock100_chapter8 as k8

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
class lesson71Test(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_stopword(self):
        self.assertEqual(True, k8.lesson71("and"))
        self.assertEqual(True, k8.lesson71("And"))
        self.assertEqual(False, k8.lesson71("stepping"))

#
# 72. 素性抽出
# 極性分析に有用そうな素性を各自で設計し，学習データから素性を抽出せよ．素性としては，レビューからストップワードを除去し，各単語をステミング処理したものが最低限のベースラインとなるであろう．
#
# 73. 学習
# 72で抽出した素性を用いて，ロジスティック回帰モデルを学習せよ．
#
# 74. 予測
# 73で学習したロジスティック回帰モデルを用い，与えられた文の極性ラベル（正例なら"+1"，負例なら"-1"）と，その予測確率を計算するプログラムを実装せよ．
#
# 75. 素性の重み
# 73で学習したロジスティック回帰モデルの中で，重みの高い素性トップ10と，重みの低い素性トップ10を確認せよ．
#
# 76. ラベル付け
# 学習データに対してロジスティック回帰モデルを適用し，正解のラベル，予測されたラベル，予測確率をタブ区切り形式で出力せよ．
#
# 77. 正解率の計測
# 76の出力を受け取り，予測の正解率，正例に関する適合率，再現率，F1スコアを求めるプログラムを作成せよ．
#
# 78. 5分割交差検定
# 76-77の実験では，学習に用いた事例を評価にも用いたため，正当な評価とは言えない．すなわち，分類器が訓練事例を丸暗記する際の性能を評価しており，モデルの汎化性能を測定していない．そこで，5分割交差検定により，極性分類の正解率，適合率，再現率，F1スコアを求めよ．
#
# 79. 適合率-再現率グラフの描画
# ロジスティック回帰モデルの分類の閾値を変化させることで，適合率-再現率グラフを描画せよ．

def main():
    unittest.main()

main()

