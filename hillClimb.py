import os
import sys
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt

from mlp import *

def climbThatHill():
	t = time.time()
	j = 10
	# Check out mlp's printParameter() for explination of this list.
	# I should plot this as a function of time.
	bestParamList = [1, 1, 20, 4, 0.01, 0.01, 10, 256, 512, 1024, 2048]  
	bestNet = mlp(trainingData, trainingTargets, bestParamList, j)
	bestNet.printParameters()
	bestError = bestNet.earlystopping(trainingData, trainingTargets, validationData, validationTargets)
	dt = 0.0
	iterations = 0
	moves = 0
	bestErrors = [bestError]
	p,m1,m2 = bestNet.confusion(testData,testTargets,True)
	confusion0 = [p]
	confusion1 = [m1]
	confusion2 = [m2]
	times = [dt]
	p,m1,m2
	paramList = bestParamList[:]
	while dt < 300:
		iterations += 1
		np.random.seed()
		for i in range(len(paramList)):
			# this tends to multiply or divide numbers by ~e
			# so it is a scale invariant smooth unimodal distribution
			paramList[i] = bestParamList[i] * math.exp(random.normalvariate(0, 0.3))
		currNet = mlp(trainingData, trainingTargets, paramList, j)
	#	currNet.printParameters()
	#	paramList
	#	currNet.setParameters(paramList)
		currError = currNet.earlystopping(trainingData, trainingTargets, validationData, validationTargets)
		dt = (time.time() - t)
		if currError < bestError:
			print "***************** MOVE ACCEPTED"
			moves += 1
			bestParamList = paramList[:]
			bestNet = currNet
			bestNet.printParameters()
			bestError = currError
			print "bestError so far: ", bestError 
			bestErrors.append(bestError)
			p,m1,m2 = bestNet.confusion(testData,testTargets,False)
			confusion0.append(p)
			confusion1.append(m1)
			confusion2.append(m2)
			times.append(dt)
	bestErrors.append(bestError)
	confusion0.append(p)
	confusion1.append(m1)
	confusion2.append(m2)
	times.append(dt)

#	plt.figure(1)
#	plt.plot(times, bestErrors)
#	plt.savefig("bestErrors.png")

#	plt.figure(2)
#	plt.plot(times, confusion0)
#	plt.savefig("P_correct.png")

#	plt.figure(3)
#	plt.plot(times, confusion1)
#	plt.savefig("worst_P_correct_|_yMax.png")

#	plt.figure(4)
#	plt.plot(times, confusion2)
#	plt.savefig("worst_P_correct_|_target.png")

	bestNet.confusion(testData,testTargets,True)
	print "Iterations: ", iterations
	if (iterations > 0):
		print "P(Move Accept): ", moves/float(iterations)
	else:
		print "P(Move Accept): -1"
	print "Best Error: ", bestError
#	bestNet.printParameters()
