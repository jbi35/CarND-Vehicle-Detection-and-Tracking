import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from skimage.feature import hog
from scipy.ndimage.measurements import label

class CarDetector:
    def __init__(self):
        # load classifier from file
        dist_pickle = pickle.load( open("svc_pickle.pickle", "rb" ) )
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle['scaler']
        self.color_space = dist_pickle["color_space"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.hog_channel = dist_pickle["hog_channel"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.spatial_feat = dist_pickle["spatial_feat"]
        self.hist_feat = dist_pickle["hist_feat"]
        self.hog_feat = dist_pickle["hog_feat"]

        self.old_heat5 = np.zeros((720,1280))
        self.old_heat4 = np.zeros((720,1280))
        self.old_heat3 = np.zeros((720,1280))
        self.old_heat2 = np.zeros((720,1280))
        self.old_heat1 = np.zeros((720,1280))

    def detect_car_in_frame(self,image,frame_id,draw=False):
        hot_windows1 = self.find_cars(image,scale=1,cells_per_step=2,ystart=400,ystop=496)
        hot_windows2 = self.find_cars(image,scale=1.5,cells_per_step=1,ystart=400,ystop=600)
        hot_windows3 = self.find_cars(image,scale=2,cells_per_step=2,ystart=400,ystop=656)
        hot_windows4 = self.find_cars(image,scale=3,cells_per_step=2,ystart=400,ystop=700)
        hot_windows = hot_windows1+hot_windows2+hot_windows3+hot_windows4

        if draw:
            hot_windows_pic1 = draw_boxes(image, hot_windows1)
            mpimg.imsave('pipeline_images/'+frame_id+'_hot_windows1.png', hot_windows_pic1)
            hot_windows_pic2 = draw_boxes(image, hot_windows2)
            mpimg.imsave('pipeline_images/'+frame_id+'_hot_windows2.png', hot_windows_pic2)
            hot_windows_pic3 = draw_boxes(image, hot_windows3)
            mpimg.imsave('pipeline_images/'+frame_id+'_hot_windows3.png', hot_windows_pic3)
            hot_windows_pic4 = draw_boxes(image, hot_windows4)
            mpimg.imsave('pipeline_images/'+frame_id+'_hot_windows4.png', hot_windows_pic4)
            hot_windows_pic = draw_boxes(image, hot_windows)
            mpimg.imsave('pipeline_images/'+frame_id+'_hot_windows_combined.png', hot_windows_pic)


        # Read in image similar to one shown above
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat,hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,2)
        #heat = threshold_heat(heat)
        self.old_heat5 = self.old_heat4
        self.old_heat4 = self.old_heat3
        self.old_heat3 = self.old_heat2
        self.old_heat2 = self.old_heat1
        self.old_heat1 = heat
        new_heat = self.old_heat5 + self.old_heat4 +self.old_heat3 +self.old_heat2 +self.old_heat1

        # average heat over 5 frames
        heat_new = apply_threshold(new_heat,8)

        # Visualize the heatmap when displaying

        heatmap = np.clip(heat_new, 0, 255)

        if draw:
            plt.imsave('pipeline_images/'+frame_id+'_heatmap_combined.png',heatmap, cmap='hot')
            heatmap_single_frame = np.clip(heat, 0, 255)
            plt.imsave('pipeline_images/'+frame_id+'_heatmap.png',heatmap_single_frame, cmap='hot')
        #return heatmap
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        if draw:
            plt.imsave('pipeline_images/'+frame_id+'_labels.png',labels[0], cmap='gray')

        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        if draw:
            plt.imsave('pipeline_images/'+frame_id+'_final_detection.png',draw_img)
        return draw_img

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self,img,scale,cells_per_step,ystart,ystop):
        window_list = []

        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, self.color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell )-1
        nyblocks = (ch1.shape[0] // self.pix_per_cell )-1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell)-1
        #cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)
                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    # Append window position to list
                    window_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        return window_list


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        return img

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

    # Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def threshold_heat(heatmap):
    # Zero out pixels below the threshold
    heatmap[heatmap >= 1] = 1
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img
