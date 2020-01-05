import string

class ModelConfig(object):
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.001
        self.training_steps = 100  # 训练步数
        self.batch_size = 128  # 每一批数据的个数？
        self.display_step = 20
        # Network Parameters
        self.num_input = 28 # MNIST data input (img shape: 28*28), 每条数据的字符串长度？
        self.timesteps = 28 # timesteps
        self.num_hidden = 128 # hidden layer num of features, 隐含层个数
        self.num_classes = 10 # MNIST total classes (0-9 digits)


config = ModelConfig()