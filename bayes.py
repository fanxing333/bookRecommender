import pickle

import pandas as pd
from pandas import DataFrame
import numpy as np
from cluster import Titles
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# 统计某一个学院对某一个聚类的感兴趣程度
# 某一个学院借的书在某一个聚类中的次数
def countLikes(school_count_array: np.ndarray, cluster: Titles):
    cluster_x = cluster.titles.values
    zeros = np.zeros((len(cluster_x), 1))
    cluster_x = np.hstack((cluster_x, zeros))

    for book in school_count_array:
        index = list(np.where(cluster_x == book[0])[0])
        if index:
            cluster_x[index[0]][1] += book[1]

    return cluster_x


if __name__ == "__main__":
    first = 5
    second = 5
    # 读取簇类
    f1 = open("data/first_clusters.pickle", "rb")
    first_clusters = pickle.load(f1)
    f1.close()
    f2 = open("data/second_clusters.pickle", "rb")
    second_clusters = pickle.load(f2)
    f2.close()
    # 读取全部借阅记录
    records = pd.read_csv("data/preprocessed_records.csv")

    names = [x.name for x in first_clusters] + [x.name for x in second_clusters]
    for name in names:
        records[name] = 0
    # 如果分数大于0.3，则标为喜欢
    for index, row in records.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['score'] > 0.3:
            for i in range(first):
                exit_tag = False
                if row["title"] in first_clusters[i].titles.values:
                    records.loc[index, first_clusters[i].name] = 1

                for j in range(second):
                    if row["title"] in second_clusters[i * second + j].titles.values:
                        records.loc[index, second_clusters[i * second + j].name] = 1
                        exit_tag = True
                        break

                if exit_tag:
                    break

    # 筛选出可用属性
    data = records[["school"] + names]
    #data = pd.read_csv("data/bayesian_network_data.csv")

    nodes = []
    for i, val in enumerate(names[first:]):
        nodes.append(('school', val))
        relation = (val, names[i//second])
        nodes.append(relation)

    model = BayesianModel(nodes)
    print(model.edges)

    # 最大似然估计
    mle = MaximumLikelihoodEstimator(model=model, data=data)
    print(mle.estimate_cpd(node="school"))

    for cpd in mle.get_parameters():
        model.add_cpds(cpd)

    infer = VariableElimination(model)
    print(model.check_model())
    model.save("data/bayesian_network.bif")
    bnm = open("data/bayesian_network.pickle", "wb")
    pickle.dump(model, bnm)
    bnm.close()





    """records = pd.read_csv("data/preprocessed_records.csv")
    rec = records[(records["school"] == "工商管理学院") | (records["school"] == "管工学院")
                  | (records["school"] == "信电学院")].copy()

    # 初始化评价类
    eva = Evaluation(rec)

    guanli_rec = rec[rec['school'] == "工商管理学院"]
    guangong_rec = rec[rec['school'] == "管工学院"]
    xindian_rec = rec[rec['school'] == "信电学院"]

    Title = Titles(titles=rec[['title']])

    # 读取停用词列表
    file_path_list = ["data/stopwords/cn_stopwords.txt", "data/stopwords/stopwords-en.txt", "data/stopwords/plus.txt"]
    Title.getStopwordsSet(file_path_list=file_path_list)

    Title.cutAndCount()
    # 返回词频最高的 top 个
    Title.getTop(dimension=50)
    # 对每一个书名 title 进行向量化
    Title.vectorize_titles()
    # 聚类
    sub_titles = Title.cluster(classCount=3)

    guanli_count = rec[rec['school'] == "工商管理学院"][["title"]]\
        .groupby("title").size().reset_index(name='counts')\
        .sort_values(by='counts', ascending=False).values

    guangong_count = rec[rec['school'] == "管工学院"][["title"]] \
        .groupby("title").size().reset_index(name='counts') \
        .sort_values(by='counts', ascending=False).values

    xindian_count = rec[rec['school'] == "信电学院"][["title"]] \
        .groupby("title").size().reset_index(name='counts') \
        .sort_values(by='counts', ascending=False).values

    schools = [guanli_count, guangong_count, xindian_count]
    schools_name = ["工商管理学院", "管工学院", "信电学院"]
    
    print("------------------------------------")
    for sub_title in sub_titles:
        allowed_titles_list = sub_title.titles.values

        for school in schools_name:
            print(eva.clusterAvgEvaluate(school, allowed_titles_list) / len(allowed_titles_list))
        rates = []
        for school in schools:
            a = countLikes(school, sub_title)
            a_count = 0
            for i in a:
                a_count += i[1]
            rates.append(a_count / len(sub_title.titles))

        for i in range(len(rates)):
            print(rates[i] / sum(rates))
        print("---------------------------")

    #sub_cluster = []
    for sub_title1 in sub_titles:
        sub_title1.cutAndCount()
        # 返回词频最高的 top 个
        sub_title1.getTop(dimension=50)
        # 对每一个书名 title 进行向量化
        sub_title1.vectorize_titles()
        # 聚类
        sub = sub_title1.cluster(classCount=2)

        #sub_cluster.append(sub)

        for sub_title in sub:
            allowed_titles_list = sub_title.titles.values
            for school in schools_name:
                print(eva.clusterAvgEvaluate(school, allowed_titles_list) / len(allowed_titles_list))
            rates = []
            for school in schools:
                a = countLikes(school, sub_title)
                a_count = 0
                for i in a:
                    a_count += i[1]
                rates.append(a_count / len(sub_title.titles))

            for i in range(len(rates)):
                print(rates[i] / sum(rates))
            print("---------------------------")"""