# ------------------------ INITIALIZATION ---------------------- #
import random
import os

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras import backend as K
import numpy   as np
import sys
from keras.datasets import mnist
from digits import *


def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()

class Switcher:
    def __init__(self):
        self.options = {
            'mnist' : lambda size, train, n_s, n, sname, tname: get_mnist(img_size=size, train=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
            'svhn'  : lambda size, train, n_s, n, sname, tname: get_svhn(img_size=size, split=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
            'usps'  : lambda size, train, n_s, n, sname, tname: get_usps(img_size=size, train=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
            'amazon': lambda size, train, n_s, n, sname, tname: get_amazon(img_size=size, train=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
            'webcam': lambda size, train, n_s, n, sname, tname: get_webcam(img_size=size, train=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
            'dslr'  : lambda size, train, n_s, n, sname, tname: get_dslr(img_size=size, train=train, n_samples=n_s, n=n, source_name=sname, tname=tname),
        }

    def get_data(self, opt,img_size=(32, 32), train=True, n_samples=False, n=1, source_name='', target_name=''):
        # path always set to default
        if opt == 'svhn': train = 'train' if train else 'test'
        return self.options[opt](img_size, train, n_samples, n, source_name, target_name)


def Create_Pairs(domain_adaptation_task,repetition,sample_per_class, source, target):
    domain_tasks = ['MNIST_to_USPS', 'MNIST_to_SVHN', 'USPS_to_MNIST', 'USPS_to_SVHN', 'SVHN_to_MNIST', 'SVHN_to_USPS', 'WEBCAM_to_AMAZON', \
    'DSLR_to_AMAZON', 'AMAZON_to_DSLR', 'AMAZON_to_WEBCAM', 'WEBCAM_to_DSLR', 'DSLR_to_WEBCAM']
    DATA = Switcher()

    UM  = domain_adaptation_task
    cc  = repetition
    SpC = sample_per_class


    if UM not in domain_tasks:
        raise Exception('domain_adaptation_task should be in ' + ' '.join(domain_tasks))


    if cc <0 or cc>10:
        raise Exception('number of repetition should be between 0 and 9.')

    if SpC <1 or SpC>7:
            raise Exception('number of sample_per_class should be between 1 and 7.')

    """
    if UM == domain_tasks[0] or UM == domain_tasks[2]:
        print('Creating pairs for repetition: '+str(cc)+' and sample_per_class: '+str(sample_per_class))

        X_train_target=np.load('./row_data/' + UM + '_X_train_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
        y_train_target=np.load('./row_data/' + UM + '_y_train_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')

        X_train_source=np.load('./row_data/' + UM + '_X_train_source_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
        y_train_source=np.load('./row_data/' + UM + '_y_train_source_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
    """
    
    # ---- source data ---- #
    X_train_source, y_train_source = DATA.get_data(opt=source.lower(), img_size=(32, 32), train=True, n_samples=False, source_name=source, target_name=target)
    X_train_source, y_train_source = np.array(X_train_source), np.array(y_train_source)

    # ---- target data ---- #
    X_train_target, y_train_target = DATA.get_data(opt=target.lower(), img_size=(32, 32), train=True, n_samples=True, n=SpC, source_name=source.lower(), target_name=target)
    X_train_target, y_train_target = np.array(X_train_target), np.array(y_train_target)

    print("source -> ", X_train_source.shape, y_train_source.shape)
    print("target -> ", X_train_target.shape, y_train_target.shape)

    Training_P=[]
    Training_N=[]

    train_source = [  ]

    print("Creating pairs!")
    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if y_train_source[trs]==y_train_target[trt]:
                Training_P.append([trs,trt])
            else:
                Training_N.append([trs,trt])


    random.shuffle(Training_N)
    Training = Training_P+Training_N[:3*len(Training_P)]
    random.shuffle(Training)


    X1=np.zeros([len(Training),32, 32],dtype='float32')
    X2=np.zeros([len(Training),32, 32],dtype='float32')

    y1=np.zeros([len(Training)])
    y2=np.zeros([len(Training)])
    yc=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:,:]=X_train_source[in1,:,:]
        X2[i,:,:]=X_train_target[in2,:,:]

        y1[i]=y_train_source[in1]
        y2[i]=y_train_target[in2]
        if y_train_source[in1]==y_train_target[in2]:
            yc[i]=1

    if not os.path.exists('pairs'):
        os.makedirs('pairs')

    print("x1 shape ", X1.shape)

    root = './data/pairs/'
    print("saving pairs\n")

    np.save(root + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X1)
    np.save(root + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X2)

    np.save(root + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y1)
    np.save(root + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y2)
    np.save(root + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', yc)


def create_model(size=(32, 32, 1)): # Model explained on paper
    model = Sequential([
        Convolution2D(6, (5, 5), input_shape=size),
        Activation('relu'),
        MaxPooling2D((2,2)),
        Convolution2D(16, (5, 5)),
        Activation('relu'),
        MaxPooling2D( (2, 2) ),
        Dropout(0.25),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def Create_Model():

    img_rows, img_cols = 16, 16
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding ='valid', # valid = no padding
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(84))
    model.add(Activation('relu'))
    return model



def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))



def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(model,domain_adaptation_task,repetition,sample_per_class, source, target, nb_classes, old_acc=0):
    print("Old acc: ", old_acc)
    domain_tasks = ['MNIST_to_USPS', 'MNIST_to_SVHN', 'USPS_to_MNIST', 'USPS_to_SVHN', 'SVHN_to_MNIST', 'SVHN_to_USPS', 'WEBCAM_to_AMAZON', \
    'DSLR_to_AMAZON', 'AMAZON_to_DSLR', 'AMAZON_to_WEBCAM', 'WEBCAM_to_DSLR', 'DSLR_to_WEBCAM']
    DATA = Switcher()

    UM = domain_adaptation_task
    cc = repetition
    SpC = sample_per_class

    if UM not in domain_tasks:
        raise Exception('domain_adaptation_task should be in ' + ' '.join(domain_tasks))

    if cc < 0 or cc > 10:
        raise Exception('number of repetition should be between 0 and 9.')

    if SpC < 1 or SpC > 7:
        raise Exception('number of sample_per_class should be between 1 and 7.')


    epoch = 80  # Epoch number
    batch_size = 256

    X_test, y_test = DATA.get_data(opt=target.lower(), img_size=(32, 32), train=False, n_samples=False, source_name=source, target_name=target)
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    if X_test.shape[1] == 1: X_test = np.moveaxis(X_test, 1, -1)
    if len(X_test.shape) == 3: X_test = np.expand_dims(X_test, -1)

    root = './data/pairs/'

    X1 = np.load(root + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    X2 = np.load(root + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    X1 = X1.reshape(X1.shape[0], 32, 32, 1)
    X2 = X2.reshape(X2.shape[0], 32, 32, 1)

    y1 = np.load(root + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    y2 = np.load(root + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    yc = np.load(root + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    y1 = np_utils.to_categorical(y1, nb_classes)
    y2 = np_utils.to_categorical(y2, nb_classes)

    print('Training the model '+domain_adaptation_task+' - Epoch '+str(epoch))
    print("shapes -> ", X1.shape, X2.shape)
    nn=batch_size
    best_Acc = 0
    all_results = []
    for e in range(epoch):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(len(y2) // nn):
            loss = model.train_on_batch([X1[i * nn:(i + 1) * nn, :, :, :], X2[i * nn:(i + 1) * nn, :, :, :]],
                                        [y1[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X2[i * nn:(i + 1) * nn, :, :, :], X1[i * nn:(i + 1) * nn, :, :, :]],
                                        [y2[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])    

        Out = model.predict([X_test, X_test])
        Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
        Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)

        if best_Acc < Acc:
            best_Acc = Acc
    print(str(e))
#    if best_Acc < old_acc: model.save( 'best_model_{}_{}shot.h5'.format(UM, SpC) )
    return best_Acc


# ------------------------------------- MAIN ------------------------------------- #

from keras.layers import Activation, Dropout, Dense
from keras.layers import Input, Lambda
from keras.models import Model
import random


def train(source, target, num_class, sample_per_class, repetitions, domain_adaptation_task):
# Creating embedding function. This corresponds to the function g in the paper.
# You may need to change the network parameters.
    model1 = create_model()

    # size of digits 16*16
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 1)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)


    # number of classes for digits classification
    nb_classes = num_class
    # digist dataset -> 10 ; office dataset -> 31

    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = .25

    # Having two streams. One for source and one for target.
    processed_a = model1(input_a)
    processed_b = model1(input_b)


    # Creating the prediction function. This corresponds to h in the paper.
    out1 = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(out1)
    out1 = Activation('softmax', name='classification')(out1)


    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='CSA')(
        [processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': contrastive_loss},
                optimizer='adadelta',
                loss_weights={'classification': 1 - alpha, 'CSA': alpha})
    initial_weights = model.get_weights()

    print('Domain Adaptation Task: ' + domain_adaptation_task)
    # let's create the positive and negative pairs using row data.
    # pairs will be saved in ./pairs directory
    #sample_per_class=1
    accs = []
    random.seed(1)
    best_acc = 0
    for repetition in range(repetitions):
        Create_Pairs(domain_adaptation_task,repetition,sample_per_class, source, target)
        Acc = training_the_model(model,domain_adaptation_task,repetition,sample_per_class, source, target, nb_classes, best_acc)
        if best_acc < Acc: best_acc = Acc

        print('Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class,repetition,Acc ))
        # reset the weights of the model
        model.set_weights(initial_weights)
        accs.append(Acc)
    #print("Accuracys: ", accs)
    print("Mean accuracy: ", np.mean(accs))
    with open('resultados_32x32.txt', 'a') as file:
        file.write( 'Task: '+domain_adaptation_task+' for '+str(sample_per_class)+'-shots:\n' )
        file.write( '\t Accuracys: {}\n'.format(accs) )
        file.write( '\t Mean accuracy: {}\n\n'.format( np.mean(accs) ) )



domains = ['mnist', 'usps', 'svhn']
office_domains = ['amazon', 'webcam', 'dslr']

# you can run the experiments for sample_per_class=1, ... , 7.
shots = list(range(1, 8))

repetitions = 10

#train('mnist', 'svhn', 10, 1, 1, 'USPS_to_MNIST') # Para probar


f = open('resultados_32x32.txt', 'a')
f.write( "\n{} DIGITS DATASET {}\n\n".format( "="*10, "="*10 ) )
f.close()

print("DIGITS DATASET")

num_class = 10
for i in range(len(domains)):
    source = domains[i]
    for j in range(len(domains)):
        target = domains[j]
        if i == j: continue
        for n in shots:
            domain_adaptation_task = '{}_to_{}'.format(source.upper(), target.upper()) 
            print("{} - shot for {}".format(n, domain_adaptation_task))
            train(source, target, num_class, n, repetitions, domain_adaptation_task)


f = open('resultados_32x32.txt', 'a')
f.write( "\n{} OFFICE DATASET {}\n\n".format( "="*10, "="*10 ) )
f.close()

print("OFFICE DATASET")

num_class = 31
for i in range(len(office_domains)):
    source = office_domains[i]
    for j in range(len(office_domains)):
        target = office_domains[j]
        if i == j: continue
        for n in shots:
            domain_adaptation_task = '{}_to_{}'.format(source.upper(), target.upper()) 
            print("{} - shot for {}".format(n, domain_adaptation_task))
            train(source, target, num_class, n, repetitions, domain_adaptation_task)


#source, target, num_class, sample_per_class, repetitions, domain_adaptation_task, file