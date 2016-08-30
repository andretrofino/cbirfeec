from clustering import *


def main():
    test_path = 'c100-test'
    centroids = generate_centroids()
    codebook = generate_codebook(centroids)

    image_filenames = load_image_paths_from_dir(test_path)

    sampled_imgs = []

    for obj in image_filenames:
        sampled = random.sample(obj, 1)
        sampled_imgs.extend(sampled)

    for img in sampled_imgs:
        print "Query Image: " + img
        print "Results:"
        print str(match(img, centroids, codebook)) + "\n"

if __name__ == "__main__":
    main()