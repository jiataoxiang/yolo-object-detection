import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToTensor

# step1: gradient magnitude and direction
def gradient_and_orientation(image): #Step1
    Ix = cv.Sobel(image,cv.CV_64F,1,0,ksize=3)
    Iy = cv.Sobel(image,cv.CV_64F,0,1,ksize=3)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix)
    # print(np.mean(magnitude))
    return magnitude, np.rad2deg(theta)

# step2: crop image
def crop_image(image, tao): 
    """
    image: input image
    tao: number of grid for each row and col
    """
    n, m = image.shape
    grid_size_r = n // tao
    if grid_size_r % 2 == 0:
        grid_size_r -= 1
    grid_size_c = m // tao
    if grid_size_c % 2 == 0:
        grid_size_c -= 1
    return image[n - grid_size_r * tao:,:grid_size_c * tao], grid_size_r, grid_size_c


# step3:form orientation histogram
def getHOG(image, magnitude, direction, grid_size_r, grid_size_c, tao):
    """
    direction: range from 0 to 180
    """
    n, m = image.shape
    HOG = np.zeros((grid_size_r, grid_size_c, 6))

    for i in range(grid_size_r):
        for j in range(grid_size_c):
            # count votes
            for r_idx in range(i * tao, (i+1)*tao):
                for c_idx in range(j * tao, (j + 1)*tao):
                    angle = direction[r_idx, c_idx]
                    if angle < 15 or 165 <= angle < 180:
                        HOG[i, j, 0] += 1
                        # HOG[i, j, 0] += magnitude[r_idx, c_idx]
                    elif 15 <= angle < 45:
                        HOG[i, j, 1] += 1
                        # HOG[i, j, 1] += magnitude[r_idx, c_idx]
                    elif 45 <= angle < 75:
                        HOG[i, j, 2] += 1
                        # HOG[i, j, 2] += magnitude[r_idx, c_idx]
                    elif 75 <= angle < 105:
                        HOG[i, j, 3] += 1
                        # HOG[i, j, 3] += magnitude[r_idx, c_idx]
                    elif 105 <= angle < 135:
                        HOG[i, j, 4] += 1
                        # HOG[i, j, 4] += magnitude[r_idx, c_idx]
                    elif 135 <= angle < 165:
                        HOG[i, j, 5] += 1
                        # HOG[i, j, 5] += magnitude[r_idx, c_idx]
    return HOG
    
# # theta = np.where(s_theta < 0, -s_theta, s_theta)
# theta = s_theta % 180
# # print(np.sum(np.where(theta < 0, 1, 0)))
# HOG_hist = getHOG(sample1, s_magnitude, theta, grid_size_r, grid_size_c, tao)

def compute_orientations():
    """
    compute unit orientation x and y vector for
    0, 30, 60, 90, 120, 150, 180
    """
    degs = [0, 30, 60, 90, 120, 150]
    orientations_x = []
    orientations_y = []
    for i in range(len(degs)):
        rad = np.deg2rad(degs[i])
        ratio = np.tan(rad)
        x = 1
        y = ratio
        norm = np.sqrt(x**2 + y**2)
        orientations_x.append(1/norm)
        orientations_y.append(ratio/norm)
    return orientations_x, orientations_y
# orientations_x, orientations_y = compute_orientations()

def plot_HOG(HOG_hist, tao, orientations_x, orientations_y):
    x, y, u, v = [], [], [], []
    n, m, k = HOG_hist.shape
    if tao % 2 == 0:
        w = tao // 2 - 0.5
    else:
        w = (tao - 1) // 2
    for i in range(n):
        for j in range(m):
            center_x = i * tao + w
            center_y = j * tao + w
            # add 6 bin
            hist = HOG_hist[i, j] # 6D-array
            for k in range(len(orientations_x)):
                x.append(center_x)
                y.append(center_y)
                votes_magnitude = hist[k % 6]
                # votes_magnitude = np.sqrt(hist[k % 6]) # This is for magnitude plot, scale down to make plot looks better
                u.append(orientations_x[k] * votes_magnitude)
                v.append(orientations_y[k] * votes_magnitude)
            for k in range(len(orientations_x)):
                x.append(center_x)
                y.append(center_y)
                votes_magnitude = hist[k % 6]
                # votes_magnitude = np.sqrt(hist[k % 6]) # This is for magnitude plot
                u.append(-orientations_x[k] * votes_magnitude)
                v.append(-orientations_y[k] * votes_magnitude)
    plt.figure(figsize=(10,10))
    plt.imshow(cropped, cmap="gray")
    # scale for occurrence
    plt.quiver(y, x, v, u, color='r', headlength=0, headwidth=0, scale=7000)

    plt.show()
    
# plot_HOG(HOG_hist, tao, orientations_x, orientations_y)

def getFlattenedHOGFeatures(image):
    # image [batch, channel, width, height]
    flattened = []
    batch_size = image.shape[0]
    channel = image.shape[1]
    for i in range(batch_size):
        for j in range(channel):
            imageLayer = image[i, j, ...].numpy()
            s_magnitude, s_theta = gradient_and_orientation(imageLayer)
            low_threshold = 20
            s_magnitude = np.where(s_magnitude < low_threshold, 0, s_magnitude)
            tao = 8
            cropped, grid_size_r, grid_size_c = crop_image(imageLayer, tao)
            theta = s_theta % 180
            HOG_hist = getHOG(imageLayer, s_magnitude, theta, grid_size_r, grid_size_c, tao)
    return torch.tensor(HOG_hist).flatten(start_dim=0, end_dim=-1)

if __name__ == "__main__":
    inputImage = Image.open("./dataset/train/images/-1x-1_jpg.rf.69d9b61e3cdb8a9047dad25099fcc8ef.jpg")
    inputImage = ToTensor()(inputImage)
    inputImage = inputImage.unsqueeze(0)
    print(inputImage.shape)
    HOG_hist = getFlattenedHOGFeatures(inputImage)
    print(HOG_hist.shape)
    # orientations_x, orientations_y = compute_orientations()
    # plot_HOG(HOG_hist, tao, orientations_x, orientations_y)