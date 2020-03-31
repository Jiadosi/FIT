import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score

import pdb

def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    #X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
    # -- ReLU -- )* MessageAll  --  output

    # dosi: extract func_level_fea from X
    # func_level_fea = tf.reshape(tf.slice(X, [0, tf.shape(X)[1]-1, 0], [-1, 1, 4]), [tf.shape(X)[0], -1])  # 取出每个Batch的矩阵的最后一行的前4列,形成一个新的矩阵，[batch, N_x] fit
    # dosi 11.5
    func_level_fea = tf.reshape(tf.slice(X, [0, tf.shape(X)[1]-1, 0], [-1, 1, -1]), [tf.shape(X)[0], -1])  # 取出每个Batch的矩阵的最后一行,形成一个新的矩阵，[batch, N_x]
    X = tf.slice(X, [0, 0, 0], [-1, tf.shape(X)[1]-1, -1])  # 取出节点特征，也就是原来的X

    node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode),
            [tf.shape(X)[0], -1, N_embed])  # 初始化所有节点的64维的向量，Uv
    
    cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
    for t in range(iter_level):
        #Message convey
        Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim] 把每个节点的所有后继节点的特征求和，放在该节点对应的行
        #Complex Function
        cur_info = tf.reshape(Li_t, [-1, N_embed])  # ? * 64
        for Wi in Wembed:  # Wi就是Pi,Wembed里面有embed_depth个P,这里就是在计算relu的那个迭代
            if (Wi == Wembed[-1]):
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))  # 
        #Adding
        tot_val_t = node_val + neigh_val_t
        #Nonlinearity
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t   #[batch, node_num, embed_dim]

    g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim] 求和，也就是把图的所有节点的特征加起来作为图的特征
    # dosi: concat 4_dim func_level features
    g_embed = tf.concat([g_embed, func_level_fea], 1)  # 把func_lebel_fea拼接到g_embed后面, [batch, embed_dim+4]

    output = tf.matmul(g_embed, W_output) + b_output  # 为什么这里要再乘一个[embed_dim, embed_dim]的随机矩阵，然后加[embed_dim, ?],据说这里是个全连接层，不用管
    return output


class graphnn(object):
    def __init__(self,
                    N_x,
                    Dtype, 
                    N_embed,
                    depth_embed,
                    N_o,
                    ITER_LEVEL,
                    lr,
                    # device = '/gpu:0'
                    device = '/cpu:0'  # dosi
                ):

        self.NODE_LABEL_DIM = N_x

        tf.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.truncated_normal(
                shape = [N_x, N_embed], stddev = 0.1, dtype = Dtype))
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.truncated_normal(
                    shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

            # W_output = tf.Variable(tf.truncated_normal(
                # shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
            # b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))

            # dosi: modify W_output, b_output to fit in new gembed
            # dosi 11.5
            W_output = tf.Variable(tf.truncated_normal(
                shape = [N_embed+11, N_o+11], stddev = 0.1, dtype = Dtype))
            b_output = tf.Variable(tf.constant(0, shape = [N_o+11], dtype = Dtype))  # 维度都加了11
            
            X1 = tf.placeholder(Dtype, [None, None, N_x]) #[B, N_node, N_x]
            msg1_mask = tf.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)  #[B, N_x]

            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
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
        config = tf.ConfigProto(device_count = {'CPU': 14})
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
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))
    
    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return cur_loss
        
    def calc_diff(self, X1, X2, mask1, mask2, X1_insts, X2_insts):  # dosi 1.13
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1:X1,
            self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
        return diff
    
    def train(self, X1, X2, mask1, mask2, y, X1_insts, X2_insts):  # dosi 1.13
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return loss
    
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
