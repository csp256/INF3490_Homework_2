import numpy as np
from sklearn.svm import SVC
import loadData as ld

def sklClassify():
	verbose = True
	trainingData, trainingTargets, validationData, validationTargets, testData, testTargets = ld.loadData('movements_day1-3.dat')
	clf = SVC()

	newTargets = []
	for i in range(len(trainingTargets)):
		newTargets.append(np.argmax(trainingTargets[i]))
	newTestTargets = []
	for i in range(len(testTargets)):
		newTestTargets.append(np.argmax(testTargets[i]))

	clf.fit(trainingData, newTargets) 

	A = np.zeros((len(testTargets[0]), len(testTargets[0])))
	for index in range(len(testTargets)):
		A[newTestTargets[index], clf.predict(testData[index])[0]] += 1
	total = sum(sum(A))
	correct = 0
	for index in range(len(testTargets[0])):
		correct += A[index,index]
	m1 = 1
	for index in range(len(testTargets[0])):
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
	for index in range(len(testTargets[0])):
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
