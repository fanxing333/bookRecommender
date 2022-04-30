from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import pickle

# 对搜索关键词进行 TopK 模糊搜索
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.models import BayesianModel

from cluster import Titles

def search(string, top=10, index = 0):
    records = pd.read_csv("data/preprocessed_records.csv")
    records['similarity'] = 0
    records = records.drop_duplicates(subset='title')[['title', 'author', 'similarity']].values

    for record in records:
        record[2] = fuzz.token_sort_ratio(string, record[0])

    return records[np.argsort(records[:, -1])][-top:][::-1]

# 判断 TopK 属于哪一子类
def getClass(TopK):
    f1 = open("data/first_clusters.pickle", "rb")
    first_clusters = pickle.load(f1)
    f1.close()
    x = {}
    for title in TopK:
        for cluster in first_clusters:
            if title[0] in cluster.titles.values:
                if cluster.name not in x.keys():
                    x[cluster.name] = 1
                else:
                    x[cluster.name] += 1
                break

    return sorted(x.items(), reverse=True)[0][0]


if __name__ == "__main__":
    # 1. 输入关键字
    myschool = "会计学院"
    string = "会计"
    # 2. 给出模糊匹配 TopK
    fuzzTopk = search(string)
    # print(fuzzTopk)
    # 3. 判断用户对哪一子类感兴趣
    cluster_class = getClass(fuzzTopk)
    #print("用户感兴趣的类别是：" + cluster_class)

    #model = BayesianModel.load('data/bayesian_network.bif', filetype='bif')
    #infer = VariableElimination(model)
    bnm = open("data/bayesian_network.pickle", "rb")
    model = pickle.load(bnm)
    bnm.close()
    infer = VariableElimination(model)
    print(model.check_model())

    # 读取簇类
    f1 = open("data/first_clusters.pickle", "rb")
    first_clusters = pickle.load(f1)
    f1.close()
    f2 = open("data/second_clusters.pickle", "rb")
    second_clusters = pickle.load(f2)
    f2.close()

    p = []
    for cc in [x.name for x in second_clusters]:
        p.append(infer.query(variables=[cc], evidence={cluster_class: 1, "school": myschool}).values[1])

    infer_cluster = second_clusters[p.index(max(p))]
    #print("用户可能感兴趣的类别是：" + infer_cluster.name)

    f3 = open("data/eva.pickle", "rb")
    sc = pickle.load(f3)
    f3.close()
    school_list = ['法学院', '食品学院', '会计学院', '外国语学院', '艺术与设计学院', '经济学院', '信电学院',
                    '统计与数学学院', '管工学院', '工商管理学院', '人文与传播学院', '旅游与城乡规划学院', '公共管理学院',
                    '金融学院', '环境学院', '马克思主义学院', '计算机与信息工程学院']
    score_list = sc[school_list.index(myschool)]
    score_list.sort_values(by=myschool, ascending=False, inplace=True)

    recommender_list = []
    for title in score_list[["title"]].values:
        if title[0] in infer_cluster.titles.values:
            recommender_list.append(title[0])
            if len(recommender_list) > 10:
                break
    print("-----模糊匹配---------")
    print("用户感兴趣的类别是：" + cluster_class)
    print(fuzzTopk)
    print("-----个性推荐---------")
    print("用户可能感兴趣的类别是：" + infer_cluster.name)
    for i in recommender_list:
        print(i)

