import os
import cv2
import random
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

number_of_images_per_objects = 72


def reshape(flat, size):
    return [flat[i:i + size] for i in range(0, len(flat), size)]


def load_pixels(dir_path, k_imgs=1.0, k_pixels=1.0, rsz=1.0):

    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        image_filenames.extend(filenames)
        break

    image_filenames = [name for name in image_filenames if name.endswith("png")]  # Skip non-image files
    image_filenames = reshape(image_filenames, number_of_images_per_objects)

    n_imgs = int(number_of_images_per_objects * k_imgs)

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
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    return centroids


def main():
    print "Loading pixels"
    pixels = load_pixels("coil-100", 0.02, 0.5, 0.5)
    print "Number of pixels: " + str(len(pixels))
    print "Clustering"
    cluster(pixels)

if __name__ == "__main__":
    main()
