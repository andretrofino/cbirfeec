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

number_of_images_per_object_train = 36

def reshape(flat, size):
    return [flat[i:i + size] for i in range(0, len(flat), size)]

def load_pixels_from_image(img_path, k_pixels = 1.0, resize = 1.0):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=resize, fy=resize)
    w, h, channels = img.shape
    total_pixels = w*h
    img = img.reshape((total_pixels, channels))
    n_pixels = int(total_pixels * k_pixels)
    sampled_pixels = random.sample(img, n_pixels)
    return sampled_pixels

def load_image_paths_from_dir(dir_path):
    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        image_filenames.extend(filenames)
        break

    image_filenames = [name for name in image_filenames if name.endswith("png")]  # Skip non-image files
    image_filenames = map(lambda img_name: os.path.join(dir_path, img_name), image_filenames)
    image_filenames = reshape(image_filenames, number_of_images_per_object_train)
    return image_filenames

def load_pixels(dir_path, k_imgs=1.0, k_pixels=1.0, resize=1.0):
    image_filenames = load_image_paths_from_dir(dir_path)

    n_imgs = int(number_of_images_per_object_train * k_imgs)

    sampled_imgs = []
    for obj in image_filenames:
        sampled = random.sample(obj, n_imgs)
        sampled_imgs.extend(sampled)

    pixels = []
    for img_path in sampled_imgs:
        sampled_pixels = load_pixels_from_image(img_path, k_pixels, resize)
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
CODEBOOK_PICKLE = 'codebook.pickle'
NUM_OF_CLUSTERS = 40

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
        pixels = load_pixels("c100-train", 0.2, 0.5, 0.5)
        print "Number of pixels: " + str(len(pixels))
        print "Clustering"
        centroids = cluster(pixels, n_clusters=NUM_OF_CLUSTERS)

        with open(CENTROIDS_PICKLE, 'wb') as handle:
              pickle.dump(centroids, handle)

    for centroid in centroids:
        print centroid

# PART 2
    print "Checking if codebook pickle exists..."
    if os.path.isfile(CODEBOOK_PICKLE):
        print "Codebook pickle found"
        with open(CODEBOOK_PICKLE, 'rb') as handle:
              codebook = pickle.load(handle)

        print codebook
    else:
        image_filenames = load_image_paths_from_dir('c100-train')
        image_filenames = [item for sublist in image_filenames for item in sublist]

        codebook = {}
        for img_path in image_filenames:
            image_pixels = load_pixels_from_image(img_path, resize = 0.25)

            pixels_centroids = np.zeros(NUM_OF_CLUSTERS)
            for pixel in image_pixels:
                centroid_index = find_closest_centroid(pixel, centroids)
                pixels_centroids[centroid_index] += 1

            codebook[img_path] = pixels_centroids
            print pixels_centroids

        with open(CODEBOOK_PICKLE, 'wb') as handle:
              pickle.dump(codebook, handle)


if __name__ == "__main__":
    main()
