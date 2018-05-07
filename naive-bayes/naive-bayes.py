#coding:utf-8
import math

class NaiveBayse():
    def __init__ (self):
        self.vocabularies = set()
        self.word_count = {}
        self.category_count = {}
        self.train_count = 0

    # TODO: word_listを受け取るのではなく文章を受け取りこっちで形態素解析とかしたい
    def train(self, word_list, category):
        for word in word_list:
            self.__count_up_word(word, category)
        self.__count_up_category (category)
        self.train_count += 1

    def __count_up_word(self, word, category):
        self.word_count.setdefault(category, {})
        self.word_count[category].setdefault(word, 0)

        self.word_count[category][word] += 1
        self.vocabularies.add(word)
    
    def __count_up_category(self, category):
        self.category_count.setdefault(category, 0)

        self.category_count[category] += 1

    def classify(self, word_list):
        best_category = None
        max_prob = -99999 # TODO: ちゃんと最小値を設定したい

        print('これから分類するよー')
        print('word_list: ', word_list)

        # それぞれのカテゴリで確率を計算し一番可能性が高いカテゴリを調べる
        for category in self.category_count.keys():
            prob = self.__score(word_list, category)
            print('category:', category, ', prob(log):', prob)
            if prob > max_prob:
                max_prob = prob
                best_category = category

        return best_category

    # 与えられるword_listに対して、categoryである確率
    def __score(self, word_list, category):
        # カテゴリー出現率
        score = math.log(self.__prior_prob(category))
        # カテゴリー内の単語出現率をすべての単語で求める
        for word in word_list:
            score += math.log(self.__word_prob(word, category)) # logなので足し算
        return score

    # 事前確率 = カテゴリー出現率
    # 学習データの対象カテゴリーの文書数　/ 学習データの文書数合計
    def __prior_prob(self, category):
        return float(self.category_count[category] / self.train_count)

    # カテゴリー内の単語出現率
    # 単語のカテゴリー内出現回数 + 1 / カテゴリー内単語数 + 学習データの全単語数 (加算スムージング)
    def __word_prob(self, word, category):
        return (self.__count_in_category(word, category) + 1.0) / (sum(self.word_count[category].values()) + len(self.vocabularies))

    # 単語のカテゴリー内出現回数
    def __count_in_category(self, word, category):
        if word in self.word_count[category]:
            return float(self.word_count[category][word])
        return 0.0

    # デバッグ用
    def status(self):
        print('==========')
        print('word_count')
        print(self.word_count)
        print('category_count')
        print(self.category_count)
        print('==========')

if __name__ == '__main__':
    nb = NaiveBayse()
    nb.train(['ボール', 'スポーツ'],'サッカー')
    nb.train(['ボール', 'スポーツ', 'バット', 'グローブ'],'野球')
    nb.train(['ボール', 'スポーツ'],'サッカー')
    nb.train(['ボール', 'スポーツ'],'サッカー')
    nb.train(['ボール', 'スポーツ', 'バット', 'グローブ'],'野球')

    # nb.status()

    result = nb.classify(['ボール', 'スポーツ', 'バット', 'グローブ'])
    print('分類結果: ', result)