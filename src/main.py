import cv2
import glob
import pickle
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import queue
from scipy.ndimage import label
from skimage.feature import hog
from moviepy.editor import VideoFileClip

class DetectionInfo():
    def __init__(self, test_images):
        self.max_size = 10
        self.old_bboxes = queue.Queue(self.max_size) 
        self.heatmap = np.zeros_like(test_images[0][:, :, 0])
        
    def get_heatmap(self):
        self.heatmap = np.zeros_like(test_images[0][:, :, 0])
        if self.old_bboxes.qsize() == self.max_size:
            for bboxes in list(self.old_bboxes.queue):
                self.heatmap = add_heat(self.heatmap, bboxes)
            self.heatmap = apply_threshold(self.heatmap, 20)
        return self.heatmap
    
    def get_labels(self):
        return label(self.get_heatmap())
        
    def add_bboxes(self, bboxes):
        if len(bboxes) < 1:
            return
        if self.old_bboxes.qsize() == self.max_size:
            self.old_bboxes.get()
        self.old_bboxes.put(bboxes)

def convert_color(img, conv='RGB2YCrCb'):
    #Converting the image from one color space to another
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'Gray':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

#Extracting hog features from an image
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    #Return 2 outputs if vis=true
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Return one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, vis_bboxes = False):
    
    draw_img = np.copy(img)
    xstart = int(img.shape[1]/5)
    xstop = img.shape[1]
    img_tosearch = img[ystart:ystop, xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            test_features = X_scaler.transform(hog_features) 
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or vis_bboxes == True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
                              
    return rectangles

def get_rectangles(image, scales = [1, 1.5, 2, 2.5, 3], 
                   ystarts = [400, 400, 450, 450, 460], 
                   ystops = [528, 550, 620, 650, 700]):
    out_rectangles = []
    for scale, ystart, ystop in zip(scales, ystarts, ystops):
        rectangles = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if len(rectangles) > 0:
            out_rectangles.append(rectangles)
    out_rectangles = [item for sublist in out_rectangles for item in sublist] 
    return out_rectangles

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    img_copy = np.copy(img)
    result_rectangles = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        area = (bbox[1][1] - bbox[0][1]) * (bbox[1][0] - bbox[0][0])
        if area > 40 * 40:
            result_rectangles.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img_copy, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return result_rectangles, img_copy

def find_vehicles_video(image, detection_info):
    bboxes = get_rectangles(image) 
    detection_info.add_bboxes(bboxes)
    labels = detection_info.get_labels()
    if len(labels) == 0:
        result_image = image
    else:
        bboxes, result_image = draw_labeled_bboxes(image,labels)

    return result_image

def find_vehicles_image(image):
    rectangles = get_rectangles(image)
    if not rectangles or len(rectangles) == 0:
        return image

    heatmap_image = np.zeros_like(image[:, :, 0])
    heatmap_image = add_heat(heatmap_image, rectangles)
    heatmap_image = apply_threshold(heatmap_image, 2)
    labels = label(heatmap_image)
    rectangles, result = draw_labeled_bboxes(image, labels)

    return result

def find_vehicles(image, input_type, detection_info=None):
    if input_type == 'image':
        return find_vehicles_image(image)
    elif input_type == 'video':
        return find_vehicles_video(image, detection_info)
    else:
        raise RuntimeError("Input type must be either image or video.")

if __name__ == '__main__':

    dist_pickle = pickle.load(open(os.path.join('..', 'data', 'svc_pickle.sav'), 'rb'))
    
    svc = dist_pickle["svc"] 
    X_scaler = dist_pickle["scaler"] 
    orient = dist_pickle["orient"] 
    pix_per_cell = dist_pickle["pix_per_cell"] 
    cell_per_block = dist_pickle["cell_per_block"] 
    spatial_size = dist_pickle["spatial_size"] 
    hist_bins = dist_pickle["hist_bins"]
    
    #Read cars and not-cars images
    #Data folders
    test_images_dir = os.path.join('..', 'test_images')

    # images are divided up into vehicles and non-vehicles
    test_images = []

    images = glob.glob(test_images_dir + '/*.jpg')

    for image in images:
        test_images.append(mpimg.imread(image))

    detection_info = DetectionInfo(test_images)
    detection_info.old_heatmap = np.zeros_like(test_images[0][:, :, 0])    

    if len(sys.argv) != 4:
        print('Not enought arguments.')
        sys.exit(1)

    input_type = sys.argv[1]
    inp = sys.argv[2]
    out = sys.argv[3]

    if input_type == 'image':
        inp = cv2.imread(inp)
        cv2.imwrite(out, find_vehicles(inp, 'image'))

    elif input_type == 'video':
        project_video = VideoFileClip(inp)
        output_clip = project_video.fl_image(lambda image: find_vehicles(image, 'video', detection_info))
        output_clip.write_videofile(out, audio=False)

    else:
        print('First argument must be either image or video.')
