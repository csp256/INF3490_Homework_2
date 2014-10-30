import os
import sys
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt


def shuffleData(movements, target):
	order = range(np.shape(movements)[0])
	np.random.shuffle(order)
	movements = movements[order,:]
	target = target[order,:]
	return movements, target

def loadData(filename):
	movements = np.loadtxt(filename,delimiter='\t')
	movements[:,:40] = movements[:,:40]-movements[:,:40].mean(axis=0)
	imax = np.concatenate((movements.max(axis=0)*np.ones((1,41)),np.abs(movements.min(axis=0)*np.ones((1,41)))),axis=0).max(axis=0)
	movements[:,:40] = movements[:,:40]/imax[:40]

	target = np.zeros((np.shape(movements)[0],8));	
	for x in range(1,9):
		indices = np.where(movements[:,40]==x)
		target[indices,x-1] = 1
		
	#np.random.seed(3) # reproducible results are important!
	#movements, target = shuffleData(movements, target)

	train = movements[::2,0:40]
	traint = target[::2]
	valid = movements[1::4,0:40]
	validt = target[1::4]
	test = movements[3::4,0:40]
	testt = target[3::4]

	return train, traint, valid, validt, test, testt

trainingData, trainingTargets, validationData, validationTargets, testData, testTargets = loadData('movements_day1-3.dat')


class mlp:
	def __init__(self, inputs, targets, paramList, j=120):
		self.i = inputs.shape[1]
		self.j = j
		self.k = targets.shape[1]

		self.beta1 = 0.1 #abs(paramList[0])
		self.beta2 = 0.1 #abs(paramList[1])
		self.eta1  = abs(paramList[2])
		self.eta2  = abs(paramList[3])
		self.lowPassV = 1 - math.exp(-abs(paramList[4])) # anticusp, initialize to ~1 +/- 0.5
		if (0.1<self.lowPassV):
			self.lowPassV = 0.1
			paramList[4] = math.log(0.1)
		self.lowPassW = 1 - math.exp(-abs(paramList[5]))
		if (0.03<self.lowPassW):
			self.lowPassW = 0.03
			paramList[5] = math.log(0.1)
		self.oneMinusLowPassV = 1 - self.lowPassV
		self.oneMinusLowPassW = 1 - self.lowPassW
		self.batchSize = 30
#		self.batchSize = int(1 + abs(paramList[6]))
#		if (256 < paramList[6]):
#			paramList[6] = 256;
		self.seed1 = 256# int(abs(paramList[7]))
		self.seed2 = 512#int(abs(paramList[8]))
		self.seed3 = 1024#int(abs(paramList[9]))
		self.seed4 = 2048#int(abs(paramList[10]))
		
#		np.random.seed(self.seed1)
		self.V = np.random.randn(self.i+1,  self.j+1)
#		np.random.seed(self.seed2)
		self.W = np.random.randn(self.j+1,  self.k)
#		np.random.seed(self.seed3)
		self.deltaV = np.zeros(self.V.shape)
		self.deltaW = np.zeros(self.W.shape)
		self.oldDeltaV = np.zeros(self.V.shape)
		self.oldDeltaW = np.zeros(self.W.shape)

		# sixth order discrete first derivative
		self.errorHistory = [0, 0, 0, 0, 0, 0, 0]
		self.finiteDifferenceCoefficients = [49.0/20.0, -6.0, 15.0/2.0, -20.0/3.0, 15.0/4.0, -6.0/5.0, 1.0/6.0] # current first, least recent last
		self.errorDerivative = 0


		
	def printParameters(self):
#		print "j        ", self.sizeOfHiddenLayer
#		print "beta1    ", self.beta1
#		print "beta2    ", self.beta2
		print "eta1     ", self.eta1
		print "eta2     ", self.eta2
		print "lowPass1 ", self.lowPassV
		print "lowPass2 ", self.lowPassW
#		print "batchSize", self.batchSize
#		print "seed1    ", self.seed1
#		print "seed2    ", self.seed2
#		print "seed3    ", self.seed3
#		print "seed4    ", self.seed4
		
		
	def padArray(self, inputData): # adds a constant -1 at end of array
		x = np.copy(inputData)
		x.resize(self.i+1)
		x[self.i] = -1
		return x
		
	def validationError(self, valid, validTargets):
		indices = valid.shape[0]
		misses = 0
		for index in range(indices):
			paddedInput = self.padArray(valid[index])
			y, yMax, a = self.forward(paddedInput)
			if yMax != np.argmax(validTargets[index]):
				misses += 1
		p,m1,m2 = self.confusion(valid,validTargets)
		#print "--"
		return ((1 - p)*(1-p)*10 + (1-m1)*(1-m1) + (1-m2)*(1-m2))
		#print misses/float(indices)
		#return misses/float(indices)
	
	def updateErrorDerivative(self):
		# I could/should use numpy here instead but its not in inner loop and is only 6 numbers so...
		self.errorDerivative = sum([a*b for a,b in zip(self.errorHistory, self.finiteDifferenceCoefficients)])
	
	def earlystopping(self, inputs, targets, valid, validTargets):
		t = time.time()
		
		if (inputs.shape[0] <= self.batchSize):
			self.batchSize = inputs.shape[0] - 1
			#print "batchSize was too large and has been changed to ", self.batchSize
		oldError = self.validationError(valid, validTargets)
		for i in range(len(self.errorHistory)):
			self.errorHistory[i] = oldError
		self.updateErrorDerivative()

		batches = 0
#		np.random.seed(self.seed4)
		# assuming we will need to train >=6 batches is reasonable and 
		# easier than handling derivative boundary conditions properly 
		while ((self.errorDerivative <= 0.000001 or batches < 6) and (time.time()-t<2)): 
			batches += 1
			inputs,targets = shuffleData(inputs,targets)
			self.deltaV = np.zeros(self.V.shape)
			self.deltaW = np.zeros(self.W.shape)
			for index in range(self.batchSize):
				paddedInput = self.padArray(inputs[index])
				y, yMax, a = self.forward(paddedInput)
				dV, dW = self.train(y, a, paddedInput, targets[index])
				self.deltaV += dV
				self.deltaW += dW
			self.deltaV /= self.batchSize # normalization
			self.deltaW /= self.batchSize
			self.deltaV = self.deltaV*self.oneMinusLowPassV + self.oldDeltaV*self.lowPassV # "momentum"
			self.deltaW = self.deltaW*self.oneMinusLowPassW + self.oldDeltaW*self.lowPassW # protip: this is not analogous to physical momentum! 
			self.oldDeltaV = np.copy(self.deltaV)
			self.oldDeltaW = np.copy(self.deltaW)
			self.deltaV *= self.eta1
			self.deltaW *= self.eta2
			self.V += self.deltaV
			self.W += self.deltaW
			
			self.errorHistory.pop()
			self.errorHistory.insert(0, self.validationError(valid, validTargets))
			self.updateErrorDerivative()
#				print "Batches performed: ", batches
#				print "Current error: ", currentError
#				print "Error derivative: ", self.errorDerivative
#				print "Error history: ", self.errorHistory
#				self.confusion(testData,testTargets,True)

		self.V -= self.deltaV # we went too far so back up a step
		self.W -= self.deltaW
		batches -= 1
		#self.confusion(testData,testTargets,True)
#		print batches, " batches of size ", self.batchSize, " completed."
		return self.validationError(valid, validTargets)

	def train(self, y, a, x, t): 
		dk = (t - y)*y*(1-y)
		dk*np.array([1,1,1,1,1,1,1,1])
		dj = a*(1-a)*self.W.dot(dk)
		return np.outer(x,dj), np.outer(a,dk) # we can multiply by eta later

	def forward(self, inputData):
		x = np.copy(inputData)
		x.resize(self.i+1)
		x[self.i] = -1

		hj = x.dot(self.V)
		a = 1/(1+np.exp(-self.beta1*hj)) 
		
		hk = a.dot(self.W)
		y = 1/(1+np.exp(-self.beta2*hk))
		return y, np.argmax(y), a


	def confusion(self, testData, testTargets, verbose=False):
		A = np.zeros((self.k, self.k))
		for index in range(testData.shape[0]):
			paddedTestDatum = self.padArray(testData[index])
			y, yMax, a = self.forward(paddedTestDatum)
			correct = np.argmax(testTargets[index])
			A[correct, yMax] += 1
		total = sum(sum(A))
		correct = 0
		for index in range(self.k):
			correct += A[index,index]
		m1 = 1
		for index in range(self.k):
			denominator = sum(A[:,index])
			if (denominator == 0):
				if (verbose):
					print "P( correct |  yMax =",index,") = NO DATA"
				m1 = -1
			else:
				v = A[index,index]/denominator
				if (v<m1):
					m1 = v
				if (verbose):
					print "P( correct |  yMax =",index,") =",v
		m2 = 1
		for index in range(self.k):
			denominator = sum(A[index,:])
			if (denominator == 0):
				if (verbose):
					print "P( correct | target=",index,") = NO DATA"
				m2 = -1
			else:
				v = A[index,index]/denominator
				if (v<m2):
					m2 = v
				if (verbose):
					print "P( correct | target=",index,") =",v
		if (verbose):
			print A
			print correct," correct of ", total
			print "P( correct ) = ", correct/total
			print "min( P( correct |  yMax  ) ) = ", m1
			print "min( P( correct | target ) ) = ", m2
		return correct/total,m1,m2



#net = mlp(trainingData, trainingTargets, paramList, 12)
#net.earlystopping(trainingData, trainingTargets, validationData, validationTargets)
#net.confusion(testData,testTargets)

from sklearn.svm import SVC
clf = SVC(C=100.0, cache_size=200, coef0=0.0, degree=3,  gamma=0.001, kernel='rbf', probability=True, shrinking=True, tol=0.001)

#newTargets = trainingTargets[:]
newTargets = []
for i in range(len(trainingTargets)):
	newTargets.append(np.argmax(trainingTargets[i]))

clf.fit(trainingData, newTargets) 
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
newTestTargets = []
for i in range(len(testTargets)):
	newTestTargets.append(np.argmax(testTargets[i]))

for i in range(len(testTargets)):
	if (int(clf.predict(testData[i])[0]) != newTestTargets[i]):
		print "SVC got case",i,"wrong."
#	print int(clf.predict(testData[i])[0]), " ", newTestTargets[i]



t = time.time()
j = 80
bestParamList = [1, 1, 200, 40, 0.001, 0.001, 50, 256, 512, 1024, 2048]  # you should plot this too...
bestNet = mlp(trainingData, trainingTargets, bestParamList, j)
bestNet.printParameters()
#bestNet.setParameters(paramsList)
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
while dt < 30:
	iterations += 1
	np.random.seed()
	for i in range(len(paramList)):
		# this tends to multiply or divide numbers by ~e
		# so it is a scale invariant smooth unimodal distribution
		paramList[i] = bestParamList[i] * math.exp(random.normalvariate(0, 1))
	currNet = mlp(trainingData, trainingTargets, paramList, j)
#	currNet.printParameters()
	paramList
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

plt.figure(1)
plt.plot(times, bestErrors)
plt.savefig("bestErrors.png")


bestNet.confusion(testData,testTargets,True)
print "Iterations: ", iterations
if (iterations > 0):
	print "P(Move Accept): ", moves/float(iterations)
else:
	print "P(Move Accept): -1"
print "Best Error: ", bestError
