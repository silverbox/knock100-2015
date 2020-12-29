import unittest
import knock100_chapter9 as k9

class lesson80Test(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_stopword(self):
        self.assertEqual(['be','undesira][ble','unnecessary','or','harmful', 'uyooo'], k9.filteredWords('be "..[]undesira][ble][[",   unnecessary, or harmful. ][][ [[]]uyooo???'))

class lesson81Test(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

def main():
    unittest.main()

main()

