# 第3章: 正規表現
# Wikipediaの記事を以下のフォーマットで書き出したファイルjawiki-country.json.gzがある．
#
#     1行に1記事の情報がJSON形式で格納される
#     各行には記事名が"title"キーに，記事本文が"text"キーの辞書オブジェクトに格納され，そのオブジェクトがJSON形式で書き出される
#     ファイル全体はgzipで圧縮される
#
# 以下の処理を行うプログラムを作成せよ．
import tarfile
import json
#
# 20. JSONデータの読み込み
# Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．問題21-29では，ここで抽出した記事本文に対して実行せよ．
def lesson20(json_data):
    for article in json_data:
        if 'イギリス' in article['text']:
            print(article['title'])
    return
#
# 21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ．
import re
def lesson21(json_data):
    repatter = re.compile('\[\[Category:.*\]\]')
    for article in json_data:
        lines = article['text'].split('\n')
        for line in lines:
            mres = repatter.search(line)
            if mres:
                print(article['title'] + ':21:' + line)
    return
#
# 22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
def lesson22(json_data):
    repatter = re.compile('\[\[Category:.*\]\]')
    for article in json_data:
        lines = article['text'].split('\n')
        for line in lines:
            mres = repatter.search(line)
            if mres:
                print(article['title'] + ':22:' + mres.group()[11:-2])
    return
#
# 23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば"== セクション名 =="なら1）を表示せよ．
def lesson23(json_data):
    repatter = re.compile('==* .* ==*')
    levpat = re.compile('==*')
    for article in json_data:
        lines = article['text'].split('\n')
        for line in lines:
            mres = repatter.search(line)
            if mres:
                lvres = levpat.search(line)
                lv = len(lvres.group()) - 1
                sec = mres.group()[lv + 2: - 2 - lv]
                print(article['title'] + ':23:' + sec + ' Lv=' + str(lv))
    return
#
# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．
def lesson24(json_data):
    repatter = re.compile('\|ファイル:.*\|.*')
    for article in json_data:
        lines = article['text'].split('\n')
        for line in lines:
            mres = repatter.search(line)
            if mres:
                print(article['title'] + ':24:' + line)
    return
#
# 25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
def lesson25(json_data):
    repatter = re.compile('^\{\{基礎情報.*')
    restapat = re.compile('\{\{.*')
    reendpat = re.compile('.*\}\}')
    re_link = re.compile('\[\[.*\]\]')
    ret = {}
    for article in json_data:
        lines = article['text'].split('\n')
        isBaseinfo = False
        env_lvl = 0
        for line in lines:
            mres = repatter.search(line)
            if mres:
                isBaseinfo = True
            if isBaseinfo:
                stares = restapat.findall(line)
                if stares:
                    env_lvl += len(stares)
                endres = reendpat.findall(line)
                if endres:
                    env_lvl -= len(endres)
                if env_lvl <= 0:
                    isBaseinfo = False
                elems = line.split('=')
                if len(elems) > 1:
                    key = elems[0][1:-1]
                    # lesson26
                    val = elems[1].replace("''''", '').replace("'''", '').replace("''", '')
                    # lesson27
                    reslinks = re_link.findall(val)
                    for reslink in reslinks:
                        val = val.replace(reslink, '')
                    ret[key] = val
                    print(article['title'] + ':24:' + key + '=' + val)
                    # lesson29
                    if '国旗画像' in key:
                        flagnm = val.replace('|', '').replace(' ', '_')
                        print('https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB:'+ flagnm)
    return ret
#
# 26. 強調マークアップの除去
# 25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ
# （参考: マークアップ早見表）．
def lesson26(json_data):
    return
#
# 27. 内部リンクの除去
# 26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ（参考: マークアップ早見表）．
def lesson27(json_data):
    return
#
# 28. MediaWikiマークアップの除去
# 27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．
def lesson28(json_data):
    return
#
# 29. 国旗画像のURLを取得する
# テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）
def lesson29(json_data):
    return

def main():
    # tar_file = tarfile.open('jawiki-country.json.gz', 'r:gz')
    f_json = open('jawiki-country.json', 'r')
    try:
        test_lines = f_json.readlines()
        json_data = []
        for line in test_lines:
            jsonobj = json.loads(line)
            json_data.append(jsonobj)
        # json_data = json.load(f_json)
        lesson20(json_data)
        lesson21(json_data)
        lesson22(json_data)
        lesson23(json_data)
        lesson24(json_data)
        lesson25(json_data)
        # lesson26(json_data, 3)
        # lesson27(json_data)
        # lesson28(json_data)
        lesson29(json_data)
    finally:
        f_json.close()
        # tar_file.close()

main()