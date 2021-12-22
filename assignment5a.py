import numpy as np
from skimage import data
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
Name: Rayhan Chowdhury


Note: After each image quantized, close the image for the next iteration to appear

Based on the results, it seems k = 4 is the elbow.

"""

pyramids = io.imread("pyramids.jpg") #image file read and stored as variable
pyramids = np.array(pyramids, dtype=np.float64)/255 #to numpy array where array will be 3 dimensional, every index being rgb values
plt.figure(1)
plt.title("Pyramids non-quantized") #base image to be quantized
plt.axis('off')
plt.imshow(pyramids) #numpy array rendered as image
plt.show()


x,y,z = pyramids.shape #x, y, z represent the shape values
quantized_pyramid = pyramids.reshape(x*y, z) #flattened for it to be quantized
print(x,y,z)
print(quantized_pyramid)




distortions = [] #inertia values
K = range(2,10) #iterating between k=2 to k=10


for k in K: #for every k in range
    km = KMeans(n_clusters=k, n_init = 10).fit(quantized_pyramid) #every iteration of every k set to 10
    labels = km.predict(quantized_pyramid)
    print("k =", k, " ", "SSE = ", km.inertia_ ) #k values and inertia values for every iteration
    distortions.append(km.inertia_)

    plt.figure(k)
    plt.axis("off")
    plt.title("Quantized at k = %i" %k)
    plt.imshow(km.cluster_centers_[labels].reshape(x,y,z)) #image reshaped back to original size and colors displayed for areas in specified clusters
    plt.show()

plt.figure(3)
plt.xlabel("k")
plt.ylabel("inertia")
plt.plot(K, distortions, 'bx-')
plt.title('k vs inertia') #identify elbow graph
plt.show()



flowers = io.imread("flower.jpg")
flowers = np.array(flowers, dtype=np.float64)/255



a,b,c = flowers.shape
quantized_flowers = flowers.reshape(a*b, c)



km = KMeans(n_clusters=4, n_init=10).fit(quantized_pyramid) #quantized image at 4 clusters since 4 was identified as the elbow
labels = km.predict(quantized_pyramid)
labels_flowers = km.predict(quantized_flowers)
plt.figure(4)
plt.axis("off")
plt.title("Quantized Image at k = 4")
plt.imshow(km.cluster_centers_[labels].reshape(x,y,z))
plt.show()
plt.figure(5)
plt.title("flower.jpg") #base second image
plt.axis('off')
plt.imshow(flowers)
plt.show()
plt.figure(6)
plt.axis("off")
plt.title("Quantized Image of Flower") #quantized of second image
plt.imshow(km.cluster_centers_[labels_flowers].reshape(a,b,c)) #image reshaped back to original size and colors displayed for areas in specified clusters for second image
plt.show()






