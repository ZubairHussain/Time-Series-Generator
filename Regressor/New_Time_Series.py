###################Generating new time series#######################

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


###Data Preprocessing###
Dataset = pd.DataFrame()
for i in range(20):
    i += 1
    filename = "competitionCsvData/sec_"+str(i)+".csv"
    Dataset = Dataset.append(pd.read_csv(filename))

#Considering ts_67 as labels to validate our model
labels = Dataset["ts_67"]

print ("length : ",len(labels))

#Dropping useless features
Dataset.drop(["Date","Open","High","Low","Close","Vwpc","Volume","ts_67"],axis=1,inplace=True)

print ("Number of features : ",len(Dataset.columns))

#print ("features : ",Dataset.columns)

# Mean_Normalizing the data with mean = 0 and variance = 1.
scaled_Dataset = preprocessing.scale(Dataset)

scaled_labels = preprocessing.scale(labels)


#print ("Mean : ",scaled_Dataset.mean(axis=0))

#print ("Standard Deviation : ",scaled_Dataset.std(axis=0))

#Splitting the data into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(scaled_Dataset, scaled_labels, test_size=0.33, random_state=42)

print ("X_train : ",X_train.shape)
print ("Y_train : ",y_train.shape)
print ("X_test : ",X_test.shape)
print ("Y_test : ",y_test.shape)



import tensorflow as tf # import the tensor flow

n_targets = 1

#Our Learning  Parameters\n",
learning_rate = 0.01
num_epochs = 5
batch_size = 100
display_step = 1

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#defining placeholders i.e. input, output and weights.
x = tf.placeholder(tf.float32,[None,67])
y = tf.placeholder(tf.float32,[None,1])
w1 = weight_variable([67,1000])
b1 = bias_variable([1000])
w2 = weight_variable([1000,500])
b2 = bias_variable([500])

out_w = weight_variable([500,n_targets])
out_b = bias_variable([n_targets])

#defining activations
affine = tf.nn.relu(tf.matmul(x,w1)+b1)

affine2 = tf.nn.relu(tf.matmul(affine,w2)+b2)

out = tf.matmul(affine2,out_w)+out_b

# define the loss function...Minimize squared errors
cost = tf.reduce_sum((tf.pow(out - y, 2))/(2 * X_train.shape[0]))

#defining optimizer to optimize the weights
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init=tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(init) # initalize\n",
    for i in range(num_epochs):
        print ("Epoch # ",i)
        batch_size=100
        nb = X_train.shape[0]/ batch_size     #number of batches
        print (int(nb))
        pre_batch =0
        for j in range(int(nb)):
            if batch_size <= X_train.shape[0]:
                xs = X_train[pre_batch:batch_size]
                ys = y_train[pre_batch:batch_size]
                pre_batch = batch_size
                batch_size += 100
                #print batch_size\n",
                ys = np.reshape(ys,(100,1))
                session.run(optimizer,feed_dict = {x:xs,y:ys})
                cost_per_batch = session.run(cost,feed_dict = {x:xs,y:ys})
                print ("Cost : ",cost_per_batch)
    
    print ("Optimization Finished!")
    saver.save(session,'model.ckpt')

    #calculating training cost
    y_train = np.reshape(y_train,(y_train.shape[0],1))
    tuning_cost = session.run(cost, feed_dict={x: X_train, y: y_train})
    train_predictions = session.run(out,feed_dict = {x: X_train})  
    print ("Tuning completed:", "cost=", "{:.9f}".format(tuning_cost))

    #calculating testing cost and getting predictions
    y_test = np.reshape(y_test,(y_test.shape[0],1))
    testing_cost = session.run(cost, feed_dict={x: X_test, y: y_test})
    test_predictions = session.run(out,feed_dict = {x: X_test})         
    print ("Testing data cost:" , testing_cost)

    predictions = train_predictions.tolist()
    predictions.extend(test_predictions.tolist())

    output = pd.DataFrame(predictions)

    #original_Dataset["predictions"] = pd.Series(predictions, index=original_Dataset.index)

    output.to_csv("output.csv")

    # Display a plot
    plt.figure()

    plt.plot(X_train, y_train, 'ro')

    plt.plot(X_test, y_test, 'go')

    W = session.run(w1)
    b = session.run(b1)

    line = np.dot(X_train,W)
    line += b

    plt.plot(X_train, line)

    plt.legend()
         
    plt.show()


    #print correct_prediction.eval({x: X_test, y: Y_test,drop_prob :1.0})
    #print "Accuracy:", accuracy.eval({x: X_test, y: Y_test,drop_prob :1.0})
