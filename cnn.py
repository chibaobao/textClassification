import functions as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
import matplotlib.pyplot as plt

class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维，变换到前一维
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # NDArray。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs
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
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    ctx = d2l.try_all_gpus() #探测是否支持GPU
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
    net.initialize(init.Xavier(), ctx=ctx)
    ## 4.1 加载词向量预训练集
    print("加载预训练的词向量，若电脑没有预训练数据集将会自动从网上下载")
    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
    print("加载与训练的词向量 ok")
    ## 4.2 初试化模型参数
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.constant_embedding.collect_params().setattr('grad_req', 'null')

    ## 5. 指定损失函数下降速度，训练轮数，并训练模型
    lr, num_epochs = 0.001, 5
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    ## loss_list ,train_acc_list ,test_acc_list ,time_list一依次对应的是损失，训练集正确率，测试集正确率，花费的时间
    ## 5.1进行训练
    loss_list ,train_acc_list ,test_acc_list ,time_list = d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
    ## 5.2 打印损失函数下降图
    plt.plot(loss_list) #损失函数下降数组
    plt.legend("loss") #可省略，图像上的字（可认为标题）设置loss
    plt.show() #显示图像
    # 6. 调用predict进行测试
    test_str = "this movie is so great" #,注意测试语句词数不能
    print(predict(net,vocab,test_str))