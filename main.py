import torch, study, random, processing, training, threshold, examples
import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import RAE
from sklearn.model_selection import train_test_split


sns.set_theme(style='darkgrid')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['1', '3', '5', '6', '8']


"""ЗАГРУЗКА ДАННЫХ"""

data = pandas.read_csv(data_2.csv)

"""ИСЛЕДОВАНИЕ ДАННЫХ"""

print('Колличество значений:', data.cid.value_counts())

ax = sns.countplot(data.cid)
ax.set_xticklabels(class_names)
cid = 1
anomaly_val = 100.1
cid_df = data[data.cid == cid].drop(labels='cid', axis=1)

"""ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ"""

normal_df = cid_df[(cid_df.value < anomaly_val)].drop(labels='number', axis=1)
anomaly_df = cid_df[(cid_df.value >= anomaly_val)].drop(labels='number', axis=1)
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=random.seed(10))
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=random.seed(10))
train_sequences = train_df.astype(np.float32).to_numpy().tolist()
val_sequences = val_df.astype(np.float32).to_numpy().tolist()
test_sequences = test_df.astype(np.float32).to_numpy().tolist()
anomaly_sequences = anomaly_df.astype(np.float32).to_numpy().tolist()
train_dataset, seq_len, n_features = processing.create_dataset(train_sequences)
val_dataset, _, _ = processing.create_dataset(val_sequences)
test_normal_dataset, _, _ = processing.create_dataset(test_sequences)
test_anomaly_dataset, _, _ = processing.create_dataset(anomaly_sequences)

"""Построение автокодера LSTM с помощью PyTorch"""

model = RAE(seq_len, n_features, device, embedding_dim=128)
model = model.to(device)

"""Обучение"""

#model, history = training.train_model(model, train_dataset, val_dataset, device, n_epochs=150)
MODEL_PATH = 'model.pth'
torch.save(model, MODEL_PATH)

"""Выбор порога обнаружения аномалий"""

_, losses = threshold.predict(model, train_dataset, device)
sns.distplot(losses)
THRESHOLD = 100.1

"""Оценка набора данных"""

predictions, pred_losses = threshold.predict(model, test_normal_dataset, device)
sns.distplot(pred_losses)
correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')
anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
predictions, pred_losses = threshold.predict(model, anomaly_dataset, device)
sns.distplot(pred_losses)
correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')
plt.show()