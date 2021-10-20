from Bert_extract import vectorize_sequences_bert
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


TRAIN_NUM = 100


data = pd.read_csv('../../data/Peerread/SOUNDNESS/sou_al-0.427-0.399.csv')

# get train/test data
x_train = data['comments'][:TRAIN_NUM]
y_train = data['SOUNDNESS'][:TRAIN_NUM]
x_test = data['comments'][TRAIN_NUM:]
y_test = data['SOUNDNESS'][TRAIN_NUM:]


# embedding
x_train = vectorize_sequences_bert(x_train,'albert-large-v2', 'albert-large-v2',300)
x_test = vectorize_sequences_bert(x_test,'albert-large-v2', 'albert-large-v2',300)

# # nb
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)

# # svm
# clf = svm.LinearSVC(max_iter=5000)
# y_pred=clf.fit(x_train, y_train).predict(x_test)

print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score(y_test, y_pred, average='macro')
conf_mat = confusion_matrix(y_test, y_pred)

print("test accuracy", accuracy)
print("precision", precision)
print("recall", recall)
print("f1score", f1score)
print(conf_mat)


