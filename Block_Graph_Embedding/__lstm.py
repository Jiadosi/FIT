import tensorflow as tf
import numpy as np

# lstm model
def lstm(max_seq_length):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(max_seq_length, 100))) # batch_size不用写
    model.add(tf.keras.layers.LSTM(100, return_state=False))
    # model.add(tf.keras.layers.Dense(3, activation='softmax'))
    # Compile the Keras model to configure the training process:
    model.compile(
        optimizer = tf.train.AdamOptimizer(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    print(model.summary())
    return model

def smodel():
    with tf.device(device):
        X1 = tf.placeholder(Dtype, [None, None, N_x])
        self.X1 = X1
        embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                        Wnode, Wembed, W_output, b_output)
        X2 = tf.placeholder(Dtype, [None, None, N_x])
        self.X2 = X2
        embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                        Wnode, Wembed, W_output, b_output)
        embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                        Wnode, Wembed, W_output, b_output)
        label = tf.placeholder(Dtype, [None, ])
        self.label = label
        self.embed1 = embed1
        
        cos = tf.reduce_sum(embed1*embed2, 1) / tf.sqrt(tf.reduce_sum(
                    embed1**2, 1) * tf.reduce_sum(embed2**2, 1) + 1e-10)
        diff = -cos
        self.diff = diff
        loss = tf.reduce_mean( (diff + label) ** 2 )
        self.loss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        self.optimizer = optimizer

    '''
    # input layer
    left_input = Input(shape=(max_seq_length,), dtype='float32')
    right_input = Input(shape=(max_seq_length,), dtype='float32')

    # embedding层
    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], \
                            input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # LSTM层
    # The 1st hidden layer
    shared_lstm_01 = LSTM(n_units_1st_layer, return_sequences=True)
    # The 2nd hidden layer
    shared_lstm_02 = LSTM(n_units_2nd_layer, activation='relu')

    left_output = shared_lstm_02(shared_lstm_01(encoded_left) )
    right_output= shared_lstm_02(shared_lstm_01(encoded_right))

    # 计算距离，输入是[left_output, right_output]
    my_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), \
                            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])


    # 整合成一个模型，输入是[left_input, right_input]，输出是[my_distance]
    # Pack it all up into a model
    smodel = Model([left_input, right_input], [my_distance])
    '''
