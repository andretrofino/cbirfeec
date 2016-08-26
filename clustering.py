import os
import cv2
import random
import pickle
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.spatial import distance

number_of_images_per_object = 72

def reshape(flat, size):
    return [flat[i:i + size] for i in range(0, len(flat), size)]

def load_pixels(dir_path, k_imgs=1.0, k_pixels=1.0, rsz=1.0):
    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        image_filenames.extend(filenames)
        break

    image_filenames = [name for name in image_filenames if name.endswith("png")]  # Skip non-image files
    image_filenames = reshape(image_filenames, number_of_images_per_object)

    n_imgs = int(number_of_images_per_object * k_imgs)

    sampled_imgs = []
    for obj in image_filenames:
        sampled = random.sample(obj, n_imgs)
        sampled_imgs.extend(sampled)

    pixels = []
    for img_name in sampled_imgs:
        img = cv2.imread(os.path.join(dir_path, img_name))
        img = cv2.resize(img, None, fx=rsz, fy=rsz)
        w, h, channels = img.shape
        total_pixels = w*h
        img = img.reshape((total_pixels, channels))
        n_pixels = int(total_pixels * k_pixels)
        sampled_pixels = random.sample(img, n_pixels)
        pixels.extend(sampled_pixels)

    return pixels

def cluster(data, n_clusters=100):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=1)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    return centroids

def find_closest_centroid(pixel, centroids):
    min_dist = sys.float_info.max
    for centroid_index, centroid_pixel in enumerate(centroids):
        dist = distance.euclidean(pixel,centroid_pixel)
        if dist < min_dist:
            min_dist = dist
            closest_centroid = centroid_index

    return closest_centroid

# Global variables
CENTROIDS_PICKLE = 'centroids.pickle'
NUM_OF_CLUSTERS = 100

def main():
# PART 1
    print "Checking if pickle exists..."
    if os.path.isfile(CENTROIDS_PICKLE):
        print "Pickle found"
        with open(CENTROIDS_PICKLE, 'rb') as handle:
              centroids = pickle.load(handle)

    else:
        print "Pickle not found"
        print "Loading pixels"
        pixels = load_pixels("coil-100", 0.1, 0.5, 0.5)
        print "Number of pixels: " + str(len(pixels))
        print "Clustering"
        centroids = cluster(pixels, n_clusters=NUM_OF_CLUSTERS)

        with open(CENTROIDS_PICKLE, 'wb') as handle:
              pickle.dump(centroids, handle)

    for centroid in centroids:
        print centroid

# PART 2
    pixels = np.array([[0,0,0],[255,255,255]])
    pixels_centroids = np.zeros(NUM_OF_CLUSTERS)
    for pixel in pixels:
        centroid_index = find_closest_centroid(pixel, centroids)
        pixels_centroids[centroid_index] += 1

    print pixels_centroids

if __name__ == "__main__":
    main()
