from datetime import datetime
import pandas as pd
import numpy as np
import pickle

class Evaluation(object):

    def __init__(self, records):
        self.records = records

    # 计算一本书的平均分数
    def bookAvgEvaluate(self, title, school=None):
        # 如果存在学院，则统计学院对这本书的平均得分
        # 如果不存在学院，则统计全校对这本书的平均得分
        if school:
            score_array = self.records[(self.records["title"] == title) & (self.records["school"] == school)]["score"].values
        else:
            score_array = self.records[self.records["title"] == title]["score"].values

        #print(score_array)
        if len(score_array) > 0:
            return [np.mean(score_array), len(score_array)]
        else:
            return [0, 0]

    # 计算一个学院在对一个集合的评价程度
    def clusterAvgEvaluate(self, school, allowed_titles):
        total_score = 0
        for title in allowed_titles:
            book_score_and_freq = self.bookAvgEvaluate(title[0], school)
            total_score += book_score_and_freq[0]*book_score_and_freq[1]
            #print(book_score_and_freq)

        return total_score

    # 计算一次借阅行为中对一本书的评分
    """def bookEvaluate(self, days):
        if days < 50:
            return round((days / 50), 2)
        else:
            return"""

    # 对records增加一列 days
    """def setDays(self):
        times_list = []
        for t in self.records[["start", "end"]].values:
            start = t[0][:10]
            end = t[1][:10]
            start = datetime.strptime(start, "%Y-%m-%d")
            end = datetime.strptime(end, "%Y-%m-%d")
            times_list.append((end - start).days)

        self.records["days"] = times_list"""

    """# 对records增加一列 score
    def setScores(self):
        evaluation = []
        for day in self.records[['days']].values:
            evaluation.append(self.bookEvaluate(day[0]))

        self.records["score"] = evaluation"""

if __name__ == "__main__":
    rec = pd.read_csv("data/preprocessed_records.csv")
    eva = Evaluation(rec)

    titles = rec.drop_duplicates(subset='title')[['title']].copy()
    schools = ["计算机与信息工程学院"]  # rec.drop_duplicates(subset='school')['school'].values
    for school in schools:
        titles[school] = 0

    for index, row in titles.iterrows():
        if index % 1000 == 0:
            print(index)
        for school in schools:
            e = eva.bookAvgEvaluate(row["title"], school)
            titles.loc[index, school] = e[0] * e[1]


    #titles.to_csv("data/titles_evaluation.csv")

    """f = open("data/dropped_titles.pickle", "rb")
    titles = pickle.load(f)
    f.close()

    print(type(titles))"""
    #school = "计算机与信息工程学院"
    #title = "计算机网络基础实验指导"
    #allowed_titles = rec[rec["school"] == school].drop_duplicates(subset='title')[["title"]].values
    #print(allowed_titles)
    #print(eva.clusterAvgEvaluate(school, allowed_titles))
    #score = eva.bookAvgEvaluate("计算机网络基础实验指导", "计算机与信息工程学院")
