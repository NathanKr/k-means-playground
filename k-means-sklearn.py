from os.path import join 
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# *********** get data
file = "ex7data2.mat"
current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,file)
mat_dict = sio.loadmat(file_name)
# print("mat_dict.keys() : ",mat_dict.keys())
X = mat_dict["X"]
x1 = X[:,0]
x2 = X[:,1]
m = x1.size

def plot(title):
    plt.title(title)
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1, x2,'x')
    plt.show()

def learn():
    # random_state equal integer number will seed the random number thus give reproduceable results
    res = KMeans(n_clusters=3, random_state=0).fit(X)
    for sample in res.cluster_centers_:
        plt.plot(sample[0],sample[1],'ro')
    
    plot('unsupervised dataset with clusters center')

# plot('unsupervised dataset')
learn()