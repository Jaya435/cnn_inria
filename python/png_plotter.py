import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', help='Name of PNG to be displayed', type=str)

def image_plot(image_path):
    img=mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    image_plot(args.image_path)
