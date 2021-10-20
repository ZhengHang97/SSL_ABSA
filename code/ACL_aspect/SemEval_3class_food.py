import tensorflow as tf
from Ladder_net_onlytop import LD
from Bert_extract import vectorize_sequences_bert
import keras
import time
import numpy as np
import pandas as pd
import pickle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

TRAIN_NUM = 100
data = pd.read_csv('../../data/SemEval/food/100-0.6200-0.5776.csv')

# print(data.label.value_counts())
# data = pd.read_csv('../../data/SemEval/SemEval14_food_no_conflict.csv')
# print(data.label.value_counts())
# df1 = data[data['label'].isin([1])]
# df1 = df1.sample(n=30, replace=False, random_state=None, axis=0)
# df2 = data[data['label'].isin([2])]
# df2 = df2.sample(n=25, replace=False, random_state=None, axis=0)
# df3 = data[data['label'].isin([3])]
# df3 = df3.sample(n=45, replace=False, random_state=None, axis=0)
# data_for_train = pd.concat([df1, df2, df3], ignore_index=True)
# data_for_train = shuffle(data_for_train)
# data = data.append(data_for_train)
# data = data.drop_duplicates(
#     subset=['text', 'AC', 'ACP', 'label'],
#     keep=False)
# df4 = data[data['label'].isin([1])]
# df4 = df4.sample(n=30, replace=False, random_state=None, axis=0)
# df5 = data[data['label'].isin([2])]
# df5 = df5.sample(n=25, replace=False, random_state=None, axis=0)
# df6 = data[data['label'].isin([3])]
# df6 = df6.sample(n=45, replace=False, random_state=None, axis=0)
# data_for_test = pd.concat([df4, df5, df6], ignore_index=True)
# data_for_test = shuffle(data_for_test)
# data = data.append(data_for_test)
# data_for_unlabel = data.drop_duplicates(subset=['text', 'AC', 'ACP', 'label'], keep=False)
# data_for_unlabel = shuffle(data_for_unlabel)
# # get train/test data
# x_train = data_for_train['text']
# y_train = data_for_train['label']
# x_test = data_for_test['text']
# y_test = data_for_test['label']
# x_train_unlabeled = data_for_unlabel['text'][:800]

x_train = data['text'][:TRAIN_NUM]
y_train = data['label'][:TRAIN_NUM]
x_test = data['text'][TRAIN_NUM:TRAIN_NUM+100]
y_test = data['label'][TRAIN_NUM:TRAIN_NUM+100]
x_train_unlabeled = data['text'][TRAIN_NUM+100:TRAIN_NUM+300]
# embedding
x_train = vectorize_sequences_bert(x_train, 'albert-large-v2', 'albert-large-v2', 1)
x_train_unlabeled = vectorize_sequences_bert(x_train_unlabeled, 'albert-large-v2', 'albert-large-v2', 1)
x_test = vectorize_sequences_bert(x_test, 'albert-large-v2', 'albert-large-v2', 1)

# label
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_train1 = np.asarray(y_train).astype('float32')
y_test1 = np.asarray(y_test).astype('float32')
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# use unlabeled data from iclr

n_rep = int(x_train_unlabeled.shape[0] / x_train.shape[0])
x_train_labeled_rep = np.concatenate([x_train] * n_rep)
y_train_labeled_rep = np.concatenate([y_train] * n_rep)




# Initialize the model

inp_size = 768
n_classes = 4

model = LD(layer_sizes=[inp_size, 768, 300, n_classes])  # 768,768,300,6
print(model.summary())


# Train the model
t1 = time.time()
model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, batch_size=32, epochs=100)
t2 = time.time()

# get metrics
y_test_pr = model.test_model.predict(x_test, batch_size=10)
accuracy = accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1))
precision = precision_score(y_test.argmax(-1), y_test_pr.argmax(-1), average='macro')
recall = recall_score(y_test.argmax(-1), y_test_pr.argmax(-1), average='macro')
f1score = f1_score(y_test.argmax(-1), y_test_pr.argmax(-1), average='macro')
conf_mat = confusion_matrix(y_test.argmax(-1), y_test_pr.argmax(-1))
elapsed_time = t2 - t1
print("test accuracy", accuracy)
print("precision", precision)
print("recall", recall)
print("f1score", f1score)
print("Elapsed time: %5.3f" % elapsed_time)
print(y_test.argmax(-1))
print('-'*100)
print(y_test_pr.argmax(-1))
print(conf_mat)
# nb
# gnb = GaussianNB()
# y_pred = gnb.fit(x_train, y_train1).predict(x_test)
#
# # svm
# clf = svm.SVC(kernel='linear',max_iter=5000)
# y_pred=clf.fit(x_train, y_train1).predict(x_test)
# #
# # print(y_pred)
# #
# #
# accuracy = accuracy_score(y_test1, y_pred)
# precision = precision_score(y_test1, y_pred, average='macro')
# recall = recall_score(y_test1, y_pred, average='macro')
# f1score = f1_score(y_test1, y_pred, average='macro')
# conf_mat = confusion_matrix(y_test1, y_pred)
#
# print("test accuracy", accuracy)
# print("precision", precision)
# print("recall", recall)
# print("f1score", f1score)
# print(conf_mat)
