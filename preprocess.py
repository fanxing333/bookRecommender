# 实验环境
# python3.8
# pandas
# numpy
# 输入: 图书馆借阅数据 records.csv
# 目的: 1. 处理数据缺失值
#      2. 合并相同学院
#      3. 对数据进行可视化分析，寻找可行方法(jupyter)
# 输出: 清理后的借阅数据 preprocessed_records.csv
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import numpy as np

# 统计各属性的缺失值情况
def count_null(records: DataFrame):
    print("school: " + str(records[records['school'].isnull()].shape[0]))
    print("class: " + str(records[records['class'].isnull()].shape[0]))
    print("start: " + str(records[records['start'].isnull()].shape[0]))
    print("end: " + str(records[records['end'].isnull()].shape[0]))
    print("call_number: " + str(records[records['call_number'].isnull()].shape[0]))
    print("title: " + str(records[records['title'].isnull()].shape[0]))
    print("author: " + str(records[records['author'].isnull()].shape[0]))
    print("isbn: " + str(records[records['isbn'].isnull()].shape[0]))

# 对records增加一列 days
def setDays(records: DataFrame):
    times_list = []
    for t in records[["start", "end"]].values:
        start = t[0][:10]
        end = t[1][:10]
        start = datetime.strptime(start, "%Y-%m-%d")
        end = datetime.strptime(end, "%Y-%m-%d")
        times_list.append((end - start).days)

    records["days"] = times_list
    return records

# 对records增加一列 score
def setScores(records):
    evaluation = []
    for day in records[['days']].values:
        evaluation.append(bookEvaluate(day[0]))

    records["score"] = evaluation
    return records

# 计算一次借阅行为中对一本书的评分
def bookEvaluate(days):

    if days < 50:
        return round((days / 50), 2)
    else:
        return 1


if __name__ == "__main__":
    records = pd.read_csv("data/records.csv")
    records = records.rename(
        columns={"学院": "school", "班级": "class", "借书时间": "start", "还书时间": "end",
                 "索书号": "call_number", "书名": "title",
                 "作者": "author", "ISBN号": "isbn"})
    count_null(records)
    # 经统计，还书时间有99个缺失值，书名有4个缺失值，作者有284个缺失值，ISBN号有836个缺失值。
    # 书名缺失即为无效值，应删除。对于ISBN号和作者的缺失可以不处理。对于还书时间的缺失值我们补充为借书时间方便后续计算。

    # 删除缺失书名的记录
    records = records.dropna(subset=["title"])
    # 补充还书时间的99个缺失值，即还书时间与借书时间一样
    records.loc[:, ["start", "end"]] = records.loc[:, ["start", "end"]].fillna(axis=1, method='ffill')
    # records # It's OK to ignore the warning
    count_null(records)

    # 由于相同的学院可能会被记录成不同的名字，这里进行一个合并。

    """
    浙江工商大学目前的学院划分为

    人文与传播学院       
    会计学院          
    信电学院          
    公共管理学院        
    外国语学院         
    工商管理学院        
    旅游与城乡规划学院     
    法学院           
    环境学院           
    管工学院          
    经济学院          
    统计与数学学院       
    艺术与设计学院        
    计算机与信息工程学院
    金融学院          
    食品学院     
    马克思主义学院
    
    1. 东语学院、日语学院、外语学院改为外国语学院
    2. 人文学院改为人文与传播学院
    3. 财会学院、财务与会计学院改为会计学院
    4. 信息学院改为管工学院
    5. 公管学院改为公共管理学院
    6. 旅游与城市管理学院、旅游学院改为旅游与城乡规划学院
    7. 管理学院改为工商管理学院
    8. 统计学院改为统计与数学学院
    9. 艺术学院改为艺术设计学院
    """

    records["school"] = np.where(records["school"] == "东语学院", "外国语学院", records["school"])
    records["school"] = np.where(records["school"] == "日语学院", "外国语学院", records["school"])
    records["school"] = np.where(records["school"] == "外语学院", "外国语学院", records["school"])
    records["school"] = np.where(records["school"] == "人文学院", "人文与传播学院", records["school"])
    records["school"] = np.where(records["school"] == "财会学院", "会计学院", records["school"])
    records["school"] = np.where(records["school"] == "财务与会计学院", "会计学院", records["school"])
    records["school"] = np.where(records["school"] == "信息学院", "管工学院", records["school"])
    records["school"] = np.where(records["school"] == "公管学院", "公共管理学院", records["school"])
    records["school"] = np.where(records["school"] == "旅游与城市管理学院", "旅游与城乡规划学院", records["school"])
    records["school"] = np.where(records["school"] == "旅游学院", "旅游与城乡规划学院", records["school"])
    records["school"] = np.where(records["school"] == "管理学院", "工商管理学院", records["school"])
    records["school"] = np.where(records["school"] == "统计学院", "统计与数学学院", records["school"])
    records["school"] = np.where(records["school"] == "艺术学院", "艺术与设计学院", records["school"])
    records["school"] = np.where(records["school"] == "艺术设计学院", "艺术与设计学院", records["school"])

    records = setDays(records)
    records = setScores(records)
    # 输出CSV文件并进行保存
    records.to_csv(r'data/preprocessed_records.csv', index=False)