import pickle
import numpy as np
from scipy.stats import truncnorm
with open("data.pkl", "br") as fh:
    data = pickle.load(fh)

img_train_N = data[0]
# img_test_N = data[1]
y_train = data[2]
# y_test = data[3]
train_labels_one_hot = data[4]
# test_labels_one_hot = data[5]
img_train = data[6]
# img_test = data[7]
image_len = 28
image_wid = 28
image_pixel = image_len*image_wid
n_percep= 10
learning_rate = 0.1
no_of_in_nodes = image_pixel
no_of_out_nodes = 10
# no_of_hidden_nodes = 500
img_train_N = img_train_N[:,1:]
with open("data1.pkl","br") as ffh:
    data1 = pickle.load(ffh)
test_imgs = data1[0]
test_labels = data1[1]
img_test = test_imgs[:2000]
y_test = test_labels[:2000]

weights = [np.random]
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd,loc=mean,scale=sd)

# activation = sum(whi * input) + bias

def sigmoid(x):
    return np.e ** x / (1 + np.e ** x)
sigmoid = np.vectorize(sigmoid)
activation_function = sigmoid

class Perceptron:
    def __init__(self,no_of_out_nodes,no_of_in_nodes,learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.learning_rate = learning_rate
        self.init_weight()
        self.lstcnt = np.zeros(10)
        self.lsterr = np.zeros(10)
        self.err = None
    def init_weight(self):
        rad = 1 / np.sqrt(no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wei = X.rvs((no_of_out_nodes, no_of_in_nodes))

    def train(self,input,label):

        input = np.array(input,ndmin=2).T
        target = np.array(label,ndmin =2).T

        output1 = np.dot(self.wei,input)   #(10,1) matrix for 10 perceptrons
        output_final = sigmoid(output1)

        #calculate the difference of the target and reality
        outputdiff = target - output_final
        #weight updates

        #learning rate*(desiredoutput - realoutput) * input Logistic Regression
        temp = self.learning_rate * outputdiff
        temp = np.multiply(temp,input.T)

        self.wei += temp

    def test(self, test_vector):
        test_vector = np.array(test_vector, ndmin=2).T
        output = sigmoid(np.dot(self.wei, test_vector))

        return output


    def err_rate(self,img_test):#error rate calculation
        cnt = 0
        err = 0

        for i in range(len(img_test)):
            output = NN.test(img_test[i])
            predict = np.argmax(output)
            actual = test_labels[i]
            if predict != actual:
                err += 1
                self.lsterr[predict] = self.lsterr[predict]+1
            cnt += 1
            self.lstcnt[predict] = self.lstcnt[predict] + 1

        self.err = err/cnt*100
        return self.err_rate, self.lstcnt, self.lsterr




if __name__ == '__main__':
    NN = Perceptron(no_of_out_nodes,no_of_in_nodes,learning_rate)
    # weight = NN.init_weight()
    # call training function
    for k in range(20):
        for i in range(len(img_train_N)):
            NN.train(img_train_N[i], train_labels_one_hot[i])

    #call the test function
    NN.err_rate(img_test)
    print(f'The overall error rate is   { NN.err} %' )

    print(f"The overall accuracy is {100-NN.err} %" )
    lstcnt = NN.lstcnt
    lsterr = NN.lsterr
    each_error = (lsterr/lstcnt)*100
    for i in range(len(each_error)):
        print(f"The error rate for number {i} is {each_error[i]} %")

# test for the first 50 samples
    for i in range(50):
        output = NN.test(img_test[i])
        predict = np.argmax(output)
        actual = test_labels[i]
        print(f'output is{output},prediction is {predict},actual is {actual}')