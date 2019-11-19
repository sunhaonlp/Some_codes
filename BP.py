import numpy as np
import math

class singe_hidden_layer_module():
    def __init__(self, input_num,hidden_num,output_num):
        """
            hidden_state b_h
            output_state y_^
        """
        self.input_num=input_num # 输入向量的维度
        self.hidden_num=hidden_num # 隐层节点的数量
        self.output_num=output_num # 输出向量的维度
        self.v=np.random.rand(input_num,hidden_num) # 输入层到隐层的权重
        self.w=np.random.rand(hidden_num,output_num) # 隐层到输出层的权重
        self.r=np.random.rand(hidden_num) # 隐层的bias
        self.o=np.random.rand(output_num) # 输出层的bias

    def forward(self,input_,output_label,mode = 'train'):
        input_num=self.input_num
        hidden_num=self.hidden_num
        output_num=self.output_num
        v=self.v
        w=self.w
        r=self.r
        o=self.o

        hidden_state=[0 for i in range(hidden_num)] # 隐层节点的值
        output_state=[0 for i in range(output_num)] # 输出节点的值

        ## 隐层节点的值
        for h in range(hidden_num):
            sum = 0
            for i in range(input_num):
                sum+=input_[i]*v[i][h]
            hidden_state[h]=sigmoid(sum-r[h])

        ## 输出节点的值
        for j in range(output_num):
            sum = 0
            for h in range(hidden_num):
                sum+=hidden_state[h]*w[h][j]
            output_state[j]=sigmoid(sum-o[j])

        loss = 0
        ## 计算单个样本的loss
        for j in range(output_num):
            loss+=math.pow(output_state[j]-output_label[j],2)

        if mode == 'test':
            return loss
        else:
            return output_state,hidden_state,output_label,input_

    def backpropogation(self,output_state,hidden_state,output_label,input_,lr):
        # 计算g_j
        g=[0 for _ in range(self.output_num)]
        for j in range(self.output_num):
            g[j]=output_state[j]*(1-output_state[j])*(output_label[j]-output_state[j])

        # 计算▲w_hj
        delat_w=[[0 for _ in range(self.output_num)] for _ in range(self.hidden_num)]
        for h in range(self.hidden_num):
            for j in range(self.output_num):
                delat_w[h][j]=lr*g[j]*hidden_state[h]

        # 计算▲Θ_j
        delat_Θ=[0 for _ in range(self.output_num)]
        for j in range(self.output_num):
            delat_Θ[j] = -lr*g[j]

        # 计算e_h
        e=[0 for _ in range(self.hidden_num)]
        for h in range(self.hidden_num):
            sum = 0
            for j in range(self.output_num):
                sum+=self.w[h][j]*g[j]
            e[h]=hidden_state[h]*(1-hidden_state[h])*sum

        # 计算▲v_ih
        delat_v=[[0 for _ in range(self.hidden_num)] for _ in range(self.input_num)]
        for i in range(self.input_num):
            for h in range(self.hidden_num):
                delat_v[i][h]=lr*e[h]*input_[i]

        # 计算▲r_h
        delat_r = [0 for _ in range(self.hidden_num)]
        for h in range(self.hidden_num):
            delat_r[h]=-lr*e[h]

        return delat_w,delat_Θ,delat_v,delat_r

    def step(self,delat_w,delat_Θ,delat_v,delat_r):
        #更新v_ih
        for i in range(self.input_num):
            for h in range(self.hidden_num):
                self.v[i][h]+=delat_v[i][h]

        # 更新w_hj
        for j in range(self.output_num):
            for h in range(self.hidden_num):
                self.w[h][j] += delat_w[h][j]

        # 更新Θ_j
        for j in range(self.output_num):
            self.o[j]+=delat_Θ[j]

        # 更新r_h
        for h in range(self.hidden_num):
            self.r[h] += delat_r[h]

def createDataSet():
    """
    创建测试的数据集
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    color_dict={}
    color_set=set()
    root_dict={}
    root_set=set()
    sound_dict={}
    sound_set=set()
    texture_dict={}
    texture_set=set()
    qi_dict={}
    qi_set=set()
    touch_dict={}
    touch_set=set()
    result_dict={}
    result_set=set()

    for i in range(len(dataSet)):
        color_set.add(dataSet[i][0])
        root_set.add(dataSet[i][1])
        sound_set.add(dataSet[i][2])
        texture_set.add(dataSet[i][3])
        qi_set.add(dataSet[i][4])
        touch_set.add(dataSet[i][5])
        result_set.add(dataSet[i][6])

    i = 0
    for color in color_set:
        color_dict[color] = i
        i+=1

    i = 0
    for root in root_set:
        root_dict[root] = i
        i+=1

    i = 0
    for sound in sound_set:
        sound_dict[sound] = i
        i+=1

    i = 0
    for texture in texture_set:
        texture_dict[texture] = i
        i+=1

    i = 0
    for touch in touch_set:
        touch_dict[touch] = i
        i+=1

    i = 0
    for qi in qi_set:
        qi_dict[qi] = i
        i+=1

    i = 0
    for result in result_set:
        result_dict[result] = i
        i+=1


    for i in range(len(dataSet)):
        dataSet[i][0]=color_dict[dataSet[i][0]]
        dataSet[i][1]=root_dict[dataSet[i][1]]
        dataSet[i][2]=sound_dict[dataSet[i][2]]
        dataSet[i][3]=texture_dict[dataSet[i][3]]
        dataSet[i][4]=qi_dict[dataSet[i][4]]
        dataSet[i][5]=touch_dict[dataSet[i][5]]
        dataSet[i][6]=result_dict[dataSet[i][6]]

    return dataSet

#sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

if __name__=="__main__":
    # 划分训练集和测试集
    dataSet=createDataSet()
    training_set=[]
    validation_set=[]
    training_num=[1,2,3,6,7,10,14,15,16,17]
    validation_num=[4,5,8,9,11,12,13]
    for num in training_num:
        training_set.append(dataSet[num-1])
    for num in validation_num:
        validation_set.append(dataSet[num-1])

    # 开始训练
    input_num=6
    hidden_num=4
    output_num=1
    lr = 0.01      # 学习率
    model = singe_hidden_layer_module(input_num,hidden_num,output_num)

    # 训练
    for training_example in range(len(training_num)):
        input_=training_set[training_example][:input_num]
        output_label=training_set[training_example][input_num:]
        output_state,hidden_state,output_label,input_=model.forward(input_,output_label)
        delat_w,delat_Θ,delat_v,delat_r=model.backpropogation(output_state, hidden_state, output_label, input_, lr)
        model.step(delat_w, delat_Θ, delat_v, delat_r)

    # 验证
    loss = []
    for validation_example in range(len(validation_num)):
        input_=validation_set[validation_example][:input_num]
        output_label=validation_set[validation_example][input_num:]
        loss.append(model.forward(input_,output_label,'test'))


    print("验证集的均方误差为:"+str(np.mean(loss)))
