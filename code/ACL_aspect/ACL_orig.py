import tensorflow as tf
from Ladder_net import LD
from Bert_extract import vectorize_sequences_bert
import keras
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')
tf.config.experimental_run_functions_eagerly(True)

LEN_REVIEW = 300
NUM_REVIEW = 200
TRAIN_NUM = 100
UNLABELED_NUM = 10000

# # get all labeled data
# data = pd.read_csv('../../data/Peerread/acl_result_complete.csv')

# # show the distribution
# print(data.ORIGINALITY.value_counts())  # 1:0, 2:1, 3:109, 4:56, 5:84

# df2 = data[data['ORIGINALITY'].isin([2])]
# df2 = df2.sample(n=1, replace=False, random_state=None, axis=0)
# df3 = data[data['ORIGINALITY'].isin([3])]
# df3 = df3.sample(n=46, replace=False, random_state=None, axis=0)
# df4 = data[data['ORIGINALITY'].isin([4])]
# df4 = df4.sample(n=24, replace=False, random_state=None, axis=0)
# df5 = data[data['ORIGINALITY'].isin([5])]
# df5 = df5.sample(n=29, replace=False, random_state=None, axis=0)
# data_for_train = pd.concat([df2, df3, df4, df5], ignore_index=True)
# data_for_train = shuffle(data_for_train)
# data = data.append(data_for_train)
# data_for_test = data.drop_duplicates(subset=['SUBSTANCE', 'CLARITY', 'APPROPRIATENESS', 'IMPACT', 'COMPARISON', 'ORIGINALITY', 'SOUNDNESS', 'comments'], keep=False)
# data_for_test = shuffle(data_for_test)
# data = pd.concat([data_for_train, data_for_test], ignore_index=True)

data = pd.read_csv('../../data/Peerread/ORIGINALITY/ori_al-0.380-0.352.csv')

# get train/test data
x_train = data['comments'][:TRAIN_NUM]
y_train = data['ORIGINALITY'][:TRAIN_NUM]
x_test = data['comments'][TRAIN_NUM:]
y_test = data['ORIGINALITY'][TRAIN_NUM:]

# Calculate the weights for each class so that we can balance the data
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)
# embedding
x_train = vectorize_sequences_bert(x_train, 'albert-large-v2', 'albert-large-v2', 300)
x_test = vectorize_sequences_bert(x_test, 'albert-large-v2', 'albert-large-v2', 300)



# get unlabeled data for noise encoder

with open('../../data/Peerread/iclr17-19_comments_al.pickle', 'rb') as f2:
    x_train_unlabeled = pickle.load(f2)
x_train_unlabeled = x_train_unlabeled[:UNLABELED_NUM]

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# use unlabeled data from iclr

n_rep = int(x_train_unlabeled.shape[0] / x_train.shape[0])
x_train_labeled_rep = np.concatenate([x_train] * n_rep)
y_train_labeled_rep = np.concatenate([y_train] * n_rep)

# Initialize the model

inp_size = 768
n_classes = 6

model = LD(layer_sizes=[inp_size, 768, 300, n_classes])  # 768,768,300,6
print(model.summary())

# Train the model

t1 = time.time()
model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, batch_size=32, epochs=100, class_weight=weights)
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
