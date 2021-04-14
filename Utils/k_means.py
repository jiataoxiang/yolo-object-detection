import os
import numpy as np

label_path = './train/labels/'

def kMeans(k=2, iteration=50, path=label_path):
    label_file_names = [f for f in os.listdir(label_path) if f[-4:] == '.txt']
    all_points = []
    for file_name in label_file_names:
        label = np.loadtxt(label_path + file_name)
        label = label.reshape((-1, 5))[:,3:]
        
        for i in range(label.shape[0]):
            all_points.append(label[i])
    
    centers = np.random.rand(k,2)
    for _ in range(iteration):
        grouped_points = []
        for i in range(centers.shape[0]):
            grouped_points.append([])
    
        for point in all_points:
            distances = []
            for i in range(centers.shape[0]):
                center = centers[i]
                distances.append(np.linalg.norm(point-center))
            grouped_points[distances.index(min(distances))].append(point)
        
        for i in range(centers.shape[0]):
            centers[i] = sum(grouped_points[i]) / len(grouped_points[i])
    
    return centers
            
    
    
    