import functions as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import matplotlib.pyplot as plt



class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs
def predict(net,vocab,test_str):
    # test_str = "i hat this movie a"
    test_str_in_list  = test_str.split(" ")
    print(test_str_in_list)
    return d2l.predict_sentiment(net, vocab,test_str_in_list)
if __name__ == "__main__":
    batch_size = 64
    ## 1.自动下载数据，如果数据已放到指定位置则注释这行代码
    # d2l.download_imdb(data_dir='./data')
    ## 2.从文件夹读取数据
    print("读取数据")
    train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
    print("读取数据 ok")
    ## 3.整理数据
    vocab = d2l.get_vocab_imdb(train_data)
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        *d2l.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(gdata.ArrayDataset(
        *d2l.preprocess_imdb(test_data, vocab)), batch_size)
    ## 4.指定参数并加载模型
    embed_size, num_hiddens, num_layers, ctx = 100, 200, 2, d2l.try_all_gpus()
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers) #实例化一个双向RNN
    net.initialize(init.Xavier(), ctx=ctx) #对模型进行初始化
    ## 4.1 加载词向量预训练集
    print("加载与训练的词向量，若电脑没有预训练数据集将会自动从网上下载")
    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
    print("加载与训练的词向量 ok")
    ## 4.2 初试化模型参数
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr('grad_req', 'null')

    ## 5. 指定损失函数下降速度，训练轮数，并训练模型
    lr, num_epochs = 0.01, 5
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    loss_list ,train_acc_list ,test_acc_list ,time_list = d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

    ## 5.2 打印损失函数下降图
    plt.plot(loss_list)  # 损失函数下降数组
    plt.legend("loss")  # 可省略，图像上的字（可认为标题）设置loss
    plt.show()  # 显示图像
    # 6. 调用predict进行测试
    test_str = "this movie is so great"  # ,注意测试语句词数不能
    print(predict(net, vocab, test_str))