import torch, study, random, processing, training, threshold, examples
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import RAE
from sklearn.model_selection import train_test_split


sns.set_theme(style='darkgrid')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""ЗАГРУЗКА ДАННЫХ"""

#data0(train)
#data1(test)
#train
#test
df = train.append(test)
#df = df.sample(frac=1.0)
#CLASS_NORMAL
#class_names
# Переименование последнего столбца
new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns

"""ИСЛЕДОВАНИЕ ДАННЫХ"""

ax = sns.countplot(df.target)
ax.set_xticklabels(class_names)
# plt.show()
classes = df.target.unique()
fig, axs = plt.subplots(nrows=len(classes) // 3 + 1, ncols=3, sharey=True, figsize=(10, 6))
for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df[df.target == cls] \
        .drop(labels='target', axis=1) \
        .mean(axis=0) \
        .to_numpy()
    study.plot_time_series_class(data, class_names[i], ax)
fig.delaxes(axs.flat[-1])
fig.tight_layout()

"""ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ"""

normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)  # Нормальные данные
anomaly_df = df[df.target != CLASS_NORMAL].drop(labels='target', axis=1)  # Аномальные данные
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
#sns.distplot(losses)
#THRESHOLD =

"""Оценка набора данных"""

predictions, pred_losses = threshold.predict(model, test_normal_dataset, device)
#sns.distplot(pred_losses)
correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')
anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
predictions, pred_losses = threshold.predict(model, anomaly_dataset, device)
#sns.distplot(pred_losses)
correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')

"""Примеры"""

fig, axs = plt.subplots(nrows=2, ncols=6, sharex=True, sharey=True, figsize=(16, 5))
for i, data in enumerate(test_normal_dataset[:6]):
   examples.plot_prediction(model, data, device, title='Normal', ax=axs[0, i])
for i, data in enumerate(test_anomaly_dataset[:6]):
   examples.plot_prediction(model, data, device, title='Anomaly', ax=axs[1, i])
fig.tight_layout()
#plt.show()