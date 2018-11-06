import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    t=np.dot(input,W)
    p=np.exp(t) 
    
    for i in range(p.shape[0]):
        p[i]=p[i]/np.sum(p[i])
    
    #prediction=np.argmax(p, axis=1)
    #loss=np.sum(np.log(p)*label)/-p.shape[0]
    
    #gradient=np.dot(np.transpose(input),(p-label))/p.shape[0]
    prediction=np.argmax(p, axis=1)
    loss=np.sum(np.log(p)*label)/-p.shape[0]+lamda/2*np.sum(W*W)
    gradient=lamda*W+ np.dot(np.transpose(input),(p-label))/p.shape[0]
    ############################################################################

    return loss, gradient, prediction
