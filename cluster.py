import pandas as pd
from pandas import DataFrame
import jieba
import math
import os
import numpy as np
import pickle

class Titles(object):

    def __init__(self, titles, dimension=50, stopwords_set=set()):
        self.titles = titles  # 必须是 pd 对象
        self.LINE_VECTOR = None  # 对象的 书名-向量键值对，向量化后才赋值
        self.dimension = dimension  # 向量的维度
        self.word_name = None  # 词频最高的 dimension 个词
        self.word_num = None  # 词频最高的 dimension 个词对应的次数
        self.stopwords_set = stopwords_set  # 停用词集合
        self.word_dict = {}  # 全部的 词-次数键值对
        self.child_list = []  # 子类集列表，存储 Titles 对象，聚类后赋值
        self.name = ""  # 统计词频后获得

    # 对 Titles 进行聚类, 返回含有子类Titles对象的列表
    def cluster(self, classCount: int):
        # 如果 书名-向量 字典为空的话进行向量化
        if not self.LINE_VECTOR:
            self.vectorize_titles()
        from sklearn.cluster import KMeans

        wordvector = list(self.LINE_VECTOR.values())
        wordlabels = list(self.LINE_VECTOR.keys())

        clf = KMeans(n_clusters=classCount)
        s = clf.fit(wordvector)
        labels = clf.labels_

        # 初始化 classCollects，对每个集合都置为 []
        classCollects = {}
        for i in range(classCount):
            classCollects[i] = []
        # 遍历并进行分类
        for i in range(len(wordvector)):
            classCollects[labels[i]].append(wordlabels[i])

        for sub in classCollects.values():
            df = DataFrame(sub, columns={"title": sub})  # 对子类转换为 DataFrame 对象
            t = Titles(titles=df, stopwords_set=self.stopwords_set)
            self.child_list.append(t)

        return self.child_list

    # 对每一个书名 title 进行向量化
    # 设置字典 LINE_VECTOR = {"title": array[] }
    def vectorize_titles(self):
        # 对标题记录去重
        title_set = self.titles.drop_duplicates(subset='title')['title'].values
        #LINE_VEC = []
        LINE_VECTOR = {}
        path = 'titles-set-cut.txt'
        if os.path.exists(path):
            os.remove(path)

        for title in title_set:
            vec = [0] * self.dimension  # 初始化一个指定维度的向量 默认50

            lines = jieba.cut(title)
            for word_generator in lines:
                if word_generator in self.word_name:
                    vec[self.word_name.index(word_generator)] = 1

            #line_vec = title.strip('\n') + ' ' + str(vec) + '\n'
            #LINE_VEC.append(line_vec)
            LINE_VECTOR[title] = np.array(vec)
            #with open('titles-set-cut.txt', 'a') as f:
            #    f.write(line_vec)
            self.LINE_VECTOR = LINE_VECTOR

    # jieba分词并统计词频
    def cutAndCount(self):
        word_dict = {}
        titles = self.titles.values
        for title in titles:
            string = ""
            for word_generator in jieba.cut(title[0]):
                if word_generator not in self.stopwords_set:
                    string += word_generator + " "
                    if word_generator not in word_dict.keys():
                        word_dict[word_generator] = 1
                    elif word_generator in word_dict.keys():
                        word_dict[word_generator] += 1

        self.word_dict = word_dict

    # 读取停用词列表
    def getStopwordsSet(self, file_path_list):
        for file_path in file_path_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                for word in f.readlines():
                    self.stopwords_set.add(word.strip('\n'))

    # 返回词频最高的 top 个
    def getTop(self, dimension, name_list):
        self.dimension = dimension
        Zip = zip(self.word_dict.values(), self.word_dict.keys())
        order = list(Zip)
        order.sort(reverse=True)

        word_name = []
        word_num = []

        for i in range(min(self.dimension, len(order))):
            word_name.append(order[i][1])
            word_num.append(order[i][0])

        # 设置该子类名字
        for i in range(min(self.dimension, len(order))):
            if order[i][1] not in name_list:
                self.name = order[i][1]
                name_list.append(self.name)
                break

        self.word_name = word_name
        self.word_num = word_num

        return order[:dimension]

# name_list = ["Alice", "Bob", "Carl", "David", "Emma", "Ford", "Gail", "Henry", "Ian", "Jacob", "Kevin", "Lester", "Mary", "Noah", "Oliver", "Parker", "Queenie", "Rose", "Scott", "Terrell", "Ulysses", "Vance", "Walker", "Xenia", "York", "Zack"]

if __name__ == "__main__":
    name_list = []
    records = pd.read_csv("data/preprocessed_records.csv")
    first = 5  # 第一层聚类簇数
    second = 5  # 第二层聚类，每一子簇的聚类簇数
    first_clusters = []
    second_clusters = []

    Title = Titles(titles=records.drop_duplicates(subset='title')[['title']])
    # 读取停用词列表
    file_path_list = ["data/stopwords/cn_stopwords.txt", "data/stopwords/stopwords-en.txt", "data/stopwords/plus.txt"]
    Title.getStopwordsSet(file_path_list=file_path_list)

    Title.cutAndCount()
    # 返回词频最高的 top 个
    Title.getTop(dimension=200)
    # 对每一个书名 title 进行向量化
    Title.vectorize_titles()
    # 第一层聚类
    first_clusters += Title.cluster(classCount=first)


    # 第二层聚类
    for sub_cluster in first_clusters:
        sub_cluster.cutAndCount()
        sub_cluster.getTop(dimension=50)
        sub_cluster.vectorize_titles()
        second_clusters += sub_cluster.cluster(classCount=second)

    for sc in second_clusters:
        sc.cutAndCount()
        sc.getTop(dimension=50)

    # 将第一层聚类结果写入临时文件存储
    f1 = open("data/first_clusters.pickle", "wb")
    pickle.dump(first_clusters, f1)
    f1.close()
    # 持久化
    f2 = open("data/second_clusters.pickle", "wb")
    pickle.dump(second_clusters, f2)
    f2.close()

    # 测试代码
    print("----第一层聚类------")
    print([x.name for x in first_clusters])
    print("----第二层聚类------")
    print([x.name for x in second_clusters])