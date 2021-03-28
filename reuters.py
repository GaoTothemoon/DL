from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical  #one-hot编码 也叫分类编码(categorical encoding)
from keras import models
from keras import layers
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswird = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]]) # .join 以‘ ’隔开       dict.get



x_train = vectorize_sequences(train_data)    #数据向量化
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
#
# history_dict = history.history
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']
# epochs = range(1, len(loss)+1)
#
# plt.plot(epochs, loss, 'bo', label = 'Training loss')
# plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
# plt.title('Training loss and validation loss')
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.legend()
# #plt.show()


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

result = model.evaluate(x_test,one_hot_test_labels)
print(result)

predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
