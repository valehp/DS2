# -*- coding: utf-8 -*-
#python main.py --dataset digits --source usps --target mnist  --steps 50000 --num 1
#python main.py --dataset digits --source mnist --target usps  --steps 1 --num 1
#python main.py --dataset digits --source mnist --target svhn  --steps 1 --num 1

import os

def ejecutar(dataset, source, target, epochs, shots):
	print( "-"*10 , source.upper(), " to ", target.upper(), "-"*10 )
	for n in shots:
		output  =  "./MUS/{}_to_{}_{}shot_{}e.txt".format( source, target, n, epochs )
		command_newfile = "python main.py --dataset {} --source {} --target {} --steps {} --num {} > {}".format( dataset, source, target, epochs, n, output ) 
		command = "python main.py --dataset {} --source {} --target {} --steps {} --num {} >> {}".format( dataset, source, target, epochs, n, output )
		print("\n Ejecutando ", command, " \n")
		for r in range(repetitions):
			print("Repetición ", r)
			if r == 0: os.system(command_newfile)
			else: os.system(command)




dataset = 'digits'
domains = ['mnist', 'usps', 'svhn']

epochs = '500'
shots = list(range(1,8))
repetitions = 10

M, U, S = 'mnist', 'usps', 'svhn'

# ------------- M -> U ------------- #
ejecutar(dataset, M, U, epochs, shots) # seguir desde epoch 5


# ------------- U -> M ------------- #
ejecutar(dataset, U, M, epochs, shots)

# ------------- S -> U ------------- #
ejecutar(dataset, S, U, epochs, shots)

# ------------- U -> S ------------- #
ejecutar(dataset, U, S, epochs, shots)

# ------------- S -> M ------------- #
ejecutar(dataset, S, M, epochs, shots)

# ------------- M-> S ------------- #
ejecutar(dataset, M, S, epochs, shots)