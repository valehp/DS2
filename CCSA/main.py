from keras.layers import Activation, Dropout, Dense
from keras.layers import Input, Lambda
from keras.models import Model
from Initialization import *
import random


def train(source, target, num_class, sample_per_class, repetitions, domain_adaptation_task):
# Creating embedding function. This corresponds to the function g in the paper.
# You may need to change the network parameters.
    model1 = create_model()

    # size of digits 16*16
    img_rows, img_cols = 16, 16
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
        Create_Pairs(domain_adaptation_task,0 ,sample_per_class, source, target)
        Acc = training_the_model(model,domain_adaptation_task,0 ,sample_per_class, source, target, nb_classes, best_acc)
        if best_acc < Acc: best_acc = Acc

        print('Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class,repetition,Acc ))
        # reset the weights of the model
        model.set_weights(initial_weights)
        accs.append(Acc)
    #print("Accuracys: ", accs)
    print("Mean accuracy: ", np.mean(accs))
    with open('resultados_CCSA_modelpaper.txt', 'a') as file:
        file.write( 'Task: '+domain_adaptation_task+' for '+str(sample_per_class)+'-shots:\n' )
        file.write( '\t Accuracys: {}\n'.format(accs) )
        file.write( '\t Mean accuracy: {}\n\n'.format( np.mean(accs) ) )



domains = ['mnist', 'usps', 'svhn']
office_domains = ['amazon', 'webcam', 'dslr']

# you can run the experiments for sample_per_class=1, ... , 7.
shots = list(range(1, 8))

repetitions = 1
num_class = 10
reps = 1
source = 'mnist'
target = 'svhn'

for n in shots[2:]:
    domain_adaptation_task = '{}_to_{}'.format(source.upper(), target.upper()) 
    train(source, target, num_class, n, reps, domain_adaptation_task)