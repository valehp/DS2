# -*- coding: utf-8 -*-
#python main.py --dataset digits --source usps --target mnist  --steps 50000 --num 1
#python main.py --dataset digits --source mnist --target usps  --steps 1 --num 1
#python main.py --dataset digits --source mnist --target svhn  --steps 1 --num 1

import os

def ejecutar(dataset, source, target, shots):
	for n in shots:
		command = "python main.py --dataset {} --source {} --target {} --steps 50000 --num {}".format( dataset, source, target, n )
		os.system(command)




dataset = 'digits'
domains = ['mnist', 'usps', 'svhn']

epochs = '500'
shots = list(range(1,8))
repetitions = 10

M, U, S = 'mnist', 'usps', 'svhn'

# ------------- M -> U ------------- #
ejecutar(dataset, M, U, shots)


# ------------- U -> M ------------- #
#ejecutar(dataset, U, M, shots)

# ------------- S -> U ------------- #
#ejecutar(dataset, S, U, shots)

# ------------- U -> S ------------- #
#ejecutar(dataset, U, S, shots)

# ------------- S -> M ------------- #
#ejecutar(dataset, S, M, shots)

# ------------- M-> S ------------- #
#ejecutar(dataset, M, S, shots)