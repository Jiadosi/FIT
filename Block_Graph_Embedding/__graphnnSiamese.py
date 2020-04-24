import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.contrib import rnn

# import pdb

def block_embed(X_inst_embed, inst_num, inst_dim, n_hidden, name):  # X -- LSTM -- return
    X = tf.reshape(X_inst_embed, [-1, inst_num, inst_dim])  # [(batch_size*node_num), 35, 100]
    # permuting batch_size*node_num and inst_num
    X = tf.transpose(X, [1, 0, 2])
    # reshaping to (batch_size*node_num*inst_num, inst_dim)
    X = tf.reshape(X, [-1, inst_dim])
    # split to get a list of 'inst_num' tensors of shape (batch_size*node_num, inst_dim)
    X = tf.split(X, inst_num, 0)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0, name=name)  # name!!!
    outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
    ret = tf.reshape(outputs[-1], [tf.shape(X_inst_embed)[0], tf.shape(X_inst_embed)[1], n_hidden])  # outputs[-1]  # (batch_size * n_hidden)
    return ret  # [batch_size, node_num, n_hidden]


def graph_embed(X, func_level_fea, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    #X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
    # -- ReLU -- )* MessageAll  --  output

    # dosi: extract func_level_fea from X
    # func_level_fea = tf.reshape(tf.slice(X, [0, tf.shape(X)[1]-1, 0], [-1, 1, 4]), [tf.shape(X)[0], -1])  # 取出每个Batch的矩阵的最后一行的前4列,形成一个新的矩阵，[batch, N_x] fit
    # dosi 11.5
    # func_level_fea = tf.reshape(tf.slice(X, [0, tf.shape(X)[1]-1, 0], [-1, 1, -1]), [tf.shape(X)[0], -1])  # 取出每个Batch的矩阵的最后一行,形成一个新的矩阵，[batch, N_x]
    # X = tf.slice(X, [0, 0, 0], [-1, tf.shape(X)[1]-1, -1])  # 取出节点特征，也就是原来的X

    node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode),
            [tf.shape(X)[0], -1, N_embed])  # 初始化所有节点的N_embed维向量, 也就是Uv, 这一步得先变成二维矩阵去和Wnode相乘(因为Wnode是二维的吧)，把N_x变成N_embed之后，再还原成3维矩阵，node_val的shape是(batch, node_num, N_embed) 
    
    cur_msg = tf.nn.relu(node_val)   #[batch, node_num, N_embed]
    for t in range(iter_level):
        #Message convey
        Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, N_embed] 把每个节点的所有后继节点的特征求和，放在该节点对应的行
        #Complex Function
        cur_info = tf.reshape(Li_t, [-1, N_embed])  # 变成2d，(batch*node_num) * N_embed
        for Wi in Wembed:  # Wi就是Pi,Wembed里面有embed_depth个P,这里就是在计算relu的那个迭代
            if (Wi == Wembed[-1]):
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))  # 变回3d 
        #Adding
        tot_val_t = node_val + neigh_val_t  # 算法line6
        #Nonlinearity
        tot_msg_t = tf.nn.tanh(tot_val_t)  # 算法line6
        cur_msg = tot_msg_t   #[batch, node_num, N_embed], 当前batch里所有结点的特征

    g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, N_embed] 求和，也就是把图的所有节点的特征加起来作为图的特征
    # dosi: concat 11_dim func_level features
    g_embed = tf.concat([g_embed, func_level_fea], 1)  # 把func_lebel_fea拼接到g_embed后面, [batch, N_embed+11]

    output = tf.matmul(g_embed, W_output) + b_output  # 为什么这里要再乘一个[N_embed, N_o]的随机矩阵，然后每一行加了[N_o],据说这里是个全连接层，不用管, 也可以说是把N_embed转化为N_o吧
    return output  # [batch, node_num, N_o]


class graphnn(object):
    def __init__(self,
                    n_hidden,  # lstm, 128
                    N_x, # NODE_FEATURE_DIM, 11
                    Dtype,
                    N_embed, # EMBED_DIM, 128
                    depth_embed,
                    N_o,  # OUTPUT_DIM, 128
                    ITER_LEVEL,
                    lr,
                    # device = '/gpu:0'
                    device = '/cpu:0'  # dosi
                ):

        self.NODE_LABEL_DIM = N_x

        tf.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.truncated_normal(
                shape = [N_x+n_hidden, N_embed], stddev = 0.1, dtype = Dtype))  # Wnode是为了把N_x->N_embed吧，就是算法里的W1
            Wembed = []  # n层全连接神经网络
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.truncated_normal(
                    shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

            # W_output = tf.Variable(tf.truncated_normal(
                # shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
            # b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))

            # dosi: modify W_output, b_output to fit in new gembed
            W_output = tf.Variable(tf.truncated_normal(
                shape = [N_embed+11, N_o], stddev = 0.1, dtype = Dtype))
            b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))

            # block embed
            X1_inst = tf.placeholder(Dtype, [None, None, 35, 100])  # 4d??, [batch, node_num, inst_num, 100]
            self.X1_inst = X1_inst
            block_embed1 = block_embed(X1_inst, 35, 100, n_hidden, 'lstm1')  # n_hidden=128

            X2_inst = tf.placeholder(Dtype, [None, None, 35, 100])
            self.X2_inst = X2_inst
            block_embed2 = block_embed(X2_inst, 35, 100, n_hidden, 'lstm2')  # [batch, node_num, n_hidden]
            # X1和block_embed1拼接起来变成X1_new,但是要注意X1中每个函数的最后一行是函数层特征，也就是说行数会比block_embed多1!!!

            X1 = tf.placeholder(Dtype, [None, None, N_x]) #[B, node_num+1, N_x]
            msg1_mask = tf.placeholder(Dtype, [None, None, None])  #[B, node_num, node_num]
            tmp_X1 = tf.slice(X1, [0, 0, 0], [-1, tf.shape(X1)[1]-1, -1])  # 取出基本块特征矩阵 [B, node_num, N_x]
            X1_f = tf.reshape(tf.slice(X1, [0, tf.shape(X1)[1]-1, 0], [-1, 1, -1]), [tf.shape(X1)[0], -1])  # 取出每个Batch的矩阵的最后一行,形成一个新的矩阵，[batch, N_x]
            X1_new = tf.concat([tmp_X1, block_embed1], 2)  # 在第三维拼接起来, [B, node_num, N_x+n_hidden]
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1_new, X1_f,  msg1_mask, N_x+n_hidden, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)  #[B, node_num, N_o]
            
            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            tmp_X2 = tf.slice(X2, [0, 0, 0], [-1, tf.shape(X2)[1]-1, -1])  # 取出基本块特征矩阵 [B, node_num, N_x]
            X2_f = tf.reshape(tf.slice(X2, [0, tf.shape(X2)[1]-1, 0], [-1, 1, -1]), [tf.shape(X2)[0], -1])  # 取出每个Batch的矩阵的最后一行,形成一个新的矩阵，[batch, N_x]
            X2_new = tf.concat([tmp_X2, block_embed2], 2)  # 在第三维拼接起来
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2_new, X2_f, msg2_mask, N_x+n_hidden, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)

            label = tf.placeholder(Dtype, [None, ]) #same: 1; different:-1
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
    
    def say(self, string):
        print( string)
        if self.log_file != None:
            self.log_file.write(string+'\n')
    
    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto(device_count = {'CPU': 12}, intra_op_parallelism_threads=2)  # CPU cores=12, inter_op_parallelism_threads=2
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('\033[1;36m{}, model loaded from file: {}\033[0m'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('\033[1;36mTraining start @ {}\033[0m'.format(datetime.datetime.now()))
    
    def get_embed(self, X1, mask1, X1_insts):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1, self.X1_inst:X1_insts})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return cur_loss
        
    def calc_diff(self, X1, X2, mask1, mask2, X1_insts, X2_insts):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1:X1,
            self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2, self.X1_inst:X1_insts, self.X2_inst:X2_insts})
        return diff
    
    def train(self, X1, X2, mask1, mask2, y, X1_insts, X2_insts):
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y, self.X1_inst:X1_insts, self.X2_inst:X2_insts})
        return loss
    
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
