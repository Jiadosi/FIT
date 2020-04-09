
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import mnist
from tensorflow.python import keras
# from tensorflow.examples.tutorials import mnist
from keras.utils import to_categorical
# from keras.datasets import mnist
import pdb

# hyperparameters 超参数
LR = 0.001
BATCH_SIZE = 50 # 每次多少个图片
BATCH_INDEX = 0  # 用于生成训练数据
INPUT_SIZE = 28  # (img shape:28 * 28)
TIME_STEPS = 28
CELL_SIZE = 50
OUTPUT_SIZE = 10  # classes(0-9 digits)
 
# 输入
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.float32, [None, n_classes])

'''
raw_inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [711, 632, 71]
]

# padding
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding = 'post', truncating='post', maxlen=5)
print('padded_inputs', padded_inputs)
'''
'''
# masking works
masking_layer = layers.Masking()
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]),
    tf.float32)
print('unmasked_embedding', unmasked_embedding)
masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

# embedding somehow doesn't work T.T
# embedding = layers.Embedding(input_dim=4, output_dim=16, mask_zero=True)
# masked_output = embedding(padded_inputs)
# print(masked_output._keras_mask)
'''
'''
# Build a tf.keras.Sequential model and start with an embedding layer.
model = tf.keras.models.Sequential()
model.add(layers.Embedding(input_dim=4, output_dim=64, mask_zero=True))
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
for units in [64, 64]:
  model.add(layers.Dense(units, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
'''
model = tf.keras.models.Sequential([
    # Add an Embedding layer expecting input vocab of size 4, and
    # output embedding dimension of size 64.
    layers.Embedding(input_dim=4, output_dim=64, mask_zero=True),
    # Add a LSTM layer with 128 internal units.
    layers.LSTM(32),
    # Add a Dense layer with 10 units and softmax activation.
    layers.Dense(10, activation='softmax')
])

model.summary()


# Compile the Keras model to configure the training process:
model.compile(
    optimizer = tf.train.AdamOptimizer(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# train the model
# historty = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
for step in range(4001):
    X_batch = X_train[BATCH_INDEX: BATCH_SIZE+BATCH_INDEX, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_SIZE+BATCH_INDEX, :]
    print(X_batch.shape, Y_batch.shape)
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_SIZE += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_SIZE

    if step % 500 == 0:
        test_loss, test_auc = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))


# predict
# model.predict(x, batch_size=32)