import pandas as pd


# load data
data = pd.read_csv('../../data/Peerread/iclr17_decision_result.csv')
data2 = pd.read_csv('../../data/Peerread/iclr17-19_comments_result.csv')

data = data.drop_duplicates()

print(len(data2))
data2 = data2.drop_duplicates()
print(len(data2))
data3 = pd.merge(data, data2, on=['comments'])  #取交集
print(len(data3))
data2 = data2.append(data)
data2 = data2.drop_duplicates(subset=['comments'],keep=False)

print(len(data2))
