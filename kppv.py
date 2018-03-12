from mnist import MNIST
import numpy as np
import operator
import time
from skimage.feature import hog
from skimage import data, exposure

def import_data(filepath, app_length, test_length) :
	mdata = MNIST(filepath)

	app_data, app_label = mdata.load_training()
	test_data, test_label = mdata.load_testing()

	print("App length = " + str(app_length))
	print("Test length = " + str(test_length))

	app_data = np.array(app_data[0:app_length], dtype=np.float32)

	'''
	reshaped = app_data[0].reshape((28, 28))
	print(reshaped)
	fd, hog_image = hog(reshaped)
	print(hog_image)
	print(fd)
	'''

	test_data = np.array(test_data[0:test_length], dtype=np.float32)

	app_label = np.array(app_label[0:app_length], dtype=np.float32)
	test_label = np.array(test_label[0:test_length], dtype=np.float32)

	return app_data, test_data, app_label, test_label

def kppv_distances(x, y) :
    m = x.shape[0]  # x has shape (m, d)
    n = y.shape[0]  # y has shape (n, d)
    x2f = np.matrix([np.sum(np.square(x),axis=1)]*n).T
    y2f = np.matrix([np.sum(np.square(y),axis=1)]*m)
    xy = np.dot(x, y.T)
    dist = np.sqrt(x2f + y2f - 2 * xy)  # shape is (m, n)
    return dist

def kppv_predict(distance, app_label, k) :
    length = distance.shape[0]  # nombre d'instances de test
    pred_label = []

    for i in range(length):
        k_neighbors = np.argsort(distance[i,:])  # plus proches voisins de l'instance test

        class_votes = {}
        for x in range(k):
            response = app_label[k_neighbors[0,x]]
            if response in class_votes :
                class_votes[response] += 1
            else :
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        pred_label.append(sorted_votes[0][0])

    return pred_label

def evaluation_classifieur(test_label, pred_label):
    number_true = 0
    num_class = len(test_label)
    for i in range(num_class):
        if test_label[i] == pred_label[i]:
            number_true += 1

    accuracy = float(number_true) / float(num_class)
    return accuracy

'''
Script
'''

start = time.time()

app_length = 1000
test_length = 300

print("> Import data and labels")
app_data, test_data, app_label, test_label = import_data('MNIST-data', app_length, test_length)

print("> Compute distance matrix")
distance = kppv_distances(test_data, app_data)
print("distance = " + str(distance.shape))

print("> Compute prediction")
pred_label = kppv_predict(distance, app_label, 20)

print("> Evaluate classifier")
accuracy = evaluation_classifieur(test_label, pred_label)
print("Accuracy = " + str(round(accuracy, 2)))

end = time.time()

print("Time = " + str(round((end - start), 2)) + "s")
