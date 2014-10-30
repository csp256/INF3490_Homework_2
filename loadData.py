import numpy as np

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
