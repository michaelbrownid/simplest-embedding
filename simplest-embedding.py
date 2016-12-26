import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

################################
# inputs

numObjects = 16
embeddingSize = 2
batchSize = 512
learningRate = 0.01

################################
# generate data.

def gendatabalanced( num ):
    """Synthetic problem: make adjacent pairs compatible and non-adjacent
incompatible for pairs of objects. Make pair (x,y) x is always less
than y and self-pairs are not included.

    Make balance of positive and negative examples for training as
there are N^2 pairs but only 2N are compatible so there would only be
1/N postive examples if not balanced.
    """
    myx = []
    myy = []
    while len(myx)<num:
        # incompatible:
        x=0
        y=0
        while(x>=(y-1)):
            x = random.randrange(numObjects)
            y = random.randrange(numObjects)
        target = [1,0]
        myx.append([x,y])
        myy.append(target)
        assert(y-x != 1)
        # compatible
        x = random.randrange(numObjects-1)
        y = x+1
        target = [0,1]
        myx.append([x,y])
        myy.append(target)
        assert(y-x == 1)
    return( (myx, myy) )

def gendatasample( num ):
    """Synthetic problem: make adjacent pairs compatible and non-adjacent
incompatible for pairs of objects. Make pair (x,y) x is always less
than y and self-pairs are not included.

    Don't try to balance positive and negatives.
    """
    myx = []
    myy = []
    while len(myx)<num:
        x=0
        y=0
        while(x>=y):
            x = random.randrange(numObjects)
            y = random.randrange(numObjects)
        if (y-x != 1):
            target = [1,0]
        else:
            target = [0,1]            
        myx.append([x,y])
        myy.append(target)

    return( (myx, myy) )

# pick the data to be used
gendata = gendatasample

################################
# compute errors
def errors( truth, pred ):
    type = []
    TP=0
    TN=0
    FP=0
    FN=0
    for ii in range(len(truth)):
        if truth[ii][1]>0.5:
            if pred[ii][1]>0.5:
                TP+=1
                type.append("TP")
            else:
                FN+=1
                type.append("FN")
        else:
            if pred[ii][1]>0.5:
                FP+=1
                type.append("FP")
            else:
                TN+=1
                type.append("TN")
    err = float(FP+FN)/float(TP+FN+FP+TN)
    fpr = float(FP)/float(FP+TN)
    fnr = float(FN)/float(FN+TP)
    perf = "\t".join(["TP","FP","FN","TN","err","fnr","fpr"]) + "\n" + "\t".join([str(xx) for xx in (TP,FP,FN,TN,err,fnr,fpr)]) 
    return( (perf, type))
                                                                                    
################################
# The tensorflow graph

# input batchSize by two integer object numbers
inputData = tf.placeholder(tf.int32, [None,2])

# target is batchSize probability of compatible: [1-Prob(compat), Prob(compat)]
target = tf.placeholder(tf.float32, [None,2])

# embedding
#with tf.device("/cpu:0"):

# this is the matrix of embedding vectors. rows=objects. cols=embedding. random [-1,1]
embedding = tf.Variable(tf.random_uniform([numObjects, embeddingSize], -1.0, 1.0), name="embedding")

# simply pull out the corresponding embedding vectors and format into needed shape
"""
# inputsData[batchNum] = [obj0, obj1] -> inputs[batchNum] = [ x0,y0, x1,y1 ]
# this example shows is 2d embedding 
trainX [[1, 3], [2, 3], [0, 2], [0, 1]]

emap [ [[ 0.0630455   0.33276749]
  [-0.02014232  0.20366192]]

 [[ 0.39619446  0.9422543 ]
  [-0.02014232  0.20366192]]

 [[-0.11473703 -0.17519975]
  [ 0.39619446  0.9422543 ]]

 [[-0.11473703 -0.17519975]
  [ 0.0630455   0.33276749]]]

# reshape take two 2d points and make 4d vector
inputs [[ 0.0630455   0.33276749 -0.02014232  0.20366192]
 [ 0.39619446  0.9422543  -0.02014232  0.20366192]
 [-0.11473703 -0.17519975  0.39619446  0.9422543 ]
 [-0.11473703 -0.17519975  0.0630455   0.33276749]]
"""
# pull out the embedding vectors from integer id
emap = tf.nn.embedding_lookup(embedding, inputData)
# reshape to feed pair of points as single vector per datapoint
inputs = tf.reshape(emap , [-1, 2*embeddingSize]) 

# Set model to linear function to logit [1-p,p] 2d prediction vector
softmaxW = tf.Variable(tf.random_normal([2*embeddingSize,2],stddev=0.1), name="softmaxW")
softmaxB = tf.Variable(tf.random_normal([1,2], stddev=0.1), name="softmaxB")
logits = tf.matmul(inputs, softmaxW ) + softmaxB
probs = tf.nn.softmax(logits)

# cross entropy loss = \sum( -log(P prediction) of correct label)/N. Jensens E(logX)<=logE(X).
cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits, target))

# optimize
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cross)

################################
# plotting functions

#### this is the learned discriminant function
def discr(x0,y0, x1, y1):
    logist = np.dot(np.array([x0,y0, x1, y1]), mysoftmaxW) + mysoftmaxB[0]
    pp = np.exp(logist)/np.sum(np.exp(logist))
    return(pp[1])
#mysoftmaxW = np.array( [[0.450804,-0.626091],[1.027304,-1.084781],[-0.791180,1.001851],[-0.615959,0.632576]])
#mysoftmaxB = np.array([[-0.762030,0.687679]])

def plotit(myepoch):
    # draw the points themselves
    plt.plot(myembedding[:,0], myembedding[:,1], "o")
    plt.plot(myembedding[:,0], myembedding[:,1], "-")

    for ii in range(numObjects):
        plt.text(myembedding[ii,0],myembedding[ii,1],str(ii))

    # plot the postive class contours for the first 4 points in k,r,g,b colors
    cc=['k','r','g','b']
    n = 32
    x = np.linspace( np.min(myembedding[:,0]), np.max(myembedding[:,0]),n)
    y = np.linspace(np.min(myembedding[:,1]),np.max(myembedding[:,1]) ,n)
    xx,yy=np.meshgrid(x,y)
    for k in range(4):
        vals = np.zeros(shape=(n,n))
        for ii in range(len(x)):
            for jj in range(len(y)):
                # value of the discrimination function for the kth point at the grid points
                vv = discr(myembedding[k,0],myembedding[k,1],xx[ii,jj],yy[ii,jj])
                vals[ii,jj] = vv
        cs = plt.contour(xx,yy,vals,colors=cc[k])
        plt.clabel(cs)

    #### For sanity plot where the 0.5 probability point is between objects 1 and 3.
    #### step along line 0 to 1 and plot where value is closest to 0.5
    aa = myembedding[1]
    bb = myembedding[3]
    err = 99999.9
    best = [-1,-1]
    for ii in range(32):
        tp = (ii/32.0)*aa + (1.0-ii/32.0)*bb
        pp = discr(aa[0],aa[1],tp[0],tp[1])
        if abs(pp-0.5)<err:
            err = abs(pp-0.5)
            best=tp
    plt.plot(best[0],best[1],'r*',markersize=20)

    plt.savefig("simplest-embedding-%d.png" % myepoch, dpi=60)

    plt.close()

################################
# Launch the graph

sess = tf.Session()
sess.run(tf.global_variables_initializer())


################################
print "---- training"

# generate data
(trainX, trainY) = gendata(batchSize)

ofp = open("simplest-embedding.training.log","w")

# Fit all training data for a number of epochs
print >>ofp, "epoch\tcross\terr"
for epoch in range(600):
    res = sess.run([cross, optimizer], feed_dict={inputData: trainX, target: trainY})
    print epoch, res[0]

    if epoch % 3 == 0:
        results= sess.run({"cross":cross,"embedding":embedding, "softmaxW":softmaxW, "softmaxB": softmaxB, "probs":probs}, feed_dict={inputData: trainX, target: trainY})
        # pull out the variables for plotting
        myembedding= results["embedding"]
        mysoftmaxW = results["softmaxW"]
        mysoftmaxB = results["softmaxB"]
        plotit(epoch)

        myerror=errors(trainY, results["probs"])
        err = myerror[0].splitlines()[1].split("\t")[4]
        print >>ofp, "%d\t%f\t%s" % (epoch, results["cross"],err)

ofp.close()

# compute final training items after training
results= sess.run({"cross":cross,"embedding":embedding, "softmaxW":softmaxW, "softmaxB": softmaxB, "probs":probs}, feed_dict={inputData: trainX, target: trainY})

for (k,v) in results.items():
    print k, v

err = errors(trainY, results["probs"])
print "errors:\n", err[0]

# pull out the variables to use later, no more training so shouldn't change
myembedding= results["embedding"]
mysoftmaxW = results["softmaxW"]
mysoftmaxB = results["softmaxB"]

################################
print "---- testing results"

# generate new data
(testX, testY) = gendata(batchSize)

# compute items on test data for computed model
results= sess.run({"cross":cross, "probs":probs}, feed_dict={inputData: testX, target: testY})

for (k,v) in results.items():
    print k, v

err = errors(testY, results["probs"])
print "errors:\n", err[0]

################################
print "---- all pairs results"

# generate all pairs with correct labels
testX=[]
testY=[]
for ii in range(numObjects-1):
    for jj in range(ii+1, numObjects):
        testX.append([ii,jj])
        if jj==(ii+1):
            testY.append([0,1])
        else:
            testY.append([1,0])

# compute items on pair data for computed model
results= sess.run({"cross":cross,"probs":probs}, feed_dict={inputData: testX, target: testY})

for (k,v) in results.items():
    print k, v

err = errors(testY, results["probs"])
print "errors:\n", err[0]

# print the results of all pairs
# for ii in range(len(err[1])):
#         myx = myembedding[testX[ii][0]]
#         myy = myembedding[testX[ii][1]]
#         print "type", err[1][ii], "input", testX[ii], "truth", testY[ii], "pred", results["probs"][ii], "predSanity", discr(myx[0],myx[1],myy[0],myy[1])

################################

sess.close()
