import numpy as np
import mnist_loader


#h or h(a)-> activation | a -> pre-activation a=Wh(L-1) +
class Network(object):

    def __init__(self,sizes):
        #sizes -> number of neurons in respective layers of the network
        #[4,5,6]-> 3 layer network 
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]] # list of lists
        self.weights=[np.random.randn(y,x)for x,y in zip(sizes[:-1],sizes[1:])]

    
    def sigmoid(self,a):
        return 1.0/(1.0+np.exp(-a))

    def sigmoid_prime(self,a):
        return self.sigmoid(a) * (1-self.sigmoid(a))
    def feedforward(self,h):
        '''Return the ouput of the network if h_0 is the input'''
        #  a -> pre-activation h->g(a)
        for w,b in zip(self.weights,self.biases): #w and b for each layer
            #w-> dim(L+1) * dim(L) n_out, n_in  #a-> n_in,1 #b->n_in,1
            a= w@h + b 
            h=self.sigmoid(a)
        
        return h
    
    def cost_derivative(self,output_activations,y):
        #y_pred-y_true
        return (output_activations-y)
    def backprop(self,x,y):

        #a->pre-activation and h->activation
        grad_w=[np.zeros(w.shape) for w in self.weights]
        grad_b=[np.zeros(b.shape) for b in self.biases]      

        #feedforward
        activation=x
        activations=[x]

        pre_activations=[]

        for w,b in zip(self.weights,self.biases):

            pre_activation=np.dot(w,activation) + b
            pre_activations.append(pre_activation)

            activation=self.sigmoid(pre_activation)
            activations.append(activation)


        #backward pass
        #delta->grad_a = dL/d_a

        #Compute the output gradient wrt grad_a
        delta=self.cost_derivative(activations[-1],y) * self.sigmoid_prime(pre_activations[-1])
        grad_b[-1]=delta
        grad_w[-1]=np.dot(delta,activations[-2].transpose())
        for i in range(2,self.num_layers):
            #i=1 -> last layer and i=2 -> second last
            # Compute gradients wrt the layer below 
                # and then wrt the pre-activation

            pre_activation=pre_activations[-i]

            sp=self.sigmoid_prime(pre_activation)
            delta=np.dot(self.weights[-i+1].transpose(), delta) * sp

            grad_b[-i]=delta
            grad_w[-i]=np.dot(delta,activations[-i-1].transpose())
        return grad_w,grad_b


        




    def update_mini_batch(self,mini_batch,eta):
        '''update weights & biases using gd with backprop to a single mini_batch'''
            #grad_w=dL/dw
        #same dim as the w you are dealing with
        grad_w=[ np.zeros(w.shape) for w in self.weights]
        grad_b=[ np.zeros(b.shape) for b in self.biases]

        #calculate gradient wrt the whole minibatch
        for x,y in mini_batch:
            delta_grad_w,delta_grad_b=self.backprop(x,y) #wrt 1 single datapoint
            grad_b=[ gb+dgb for gb,dgb in zip(grad_b,delta_grad_b)]
            grad_w=[gw + dgw for gw,dgw in zip(grad_w,delta_grad_w)]
        
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,grad_w)]
        self.biases=[b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, grad_b)]
        #update wrt the gradient(mini_batch)


    def SGD(self,train_data, epochs, mini_batch_size, eta, test_data=None):
        '''train the neural net with stochastic mini batch gd. training data 
            & test_data are list of tuples (x,y)
            If test data is provided, the network will be evaluated against
            the test data
        '''

        if test_data is not None:
            n_test=len(test_data)
        
        n=len(train_data)

        for _  in range(epochs):
            np.random.shuffle(train_data)

            mini_batches=[ train_data[i:i+mini_batch_size] for i in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            
            if test_data:
                print(f"Epoch{_}:{self.evaluate(test_data)},{n_test}")
            else:
                print(f"Epoch{_} complete...")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return (f"{sum(int(x == y) for (x, y) in test_results) / len(test_data)  * 100} %") #returning accuracy
    

training_data, validation_data, test_data=mnist_loader.load_data_wrapper()
net=Network([784,30,10])
print(net.SGD(training_data, 30, 10, 3.0, test_data=test_data)) 