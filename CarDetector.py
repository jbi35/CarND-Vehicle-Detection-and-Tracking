import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from FeatureGenerator import single_img_features, get_hog_features, convert_color, color_hist,  bin_spatial
import cv2
import pickle
from scipy.ndimage.measurements import label
from classify_images import slide_window, search_windows, draw_boxes

class CarDetector:
    def __init__(self):
        #self.y_start_stop = [None, None] # Min and max in y to search in slide_window()
        self.y_start_stop = [400, 656]

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

    def detect_car_in_frame(self,image):
        draw_image = np.copy(image)
        
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.5, 0.5))

        hot_windows = search_windows(image, windows, self.svc, self.X_scaler, color_space=self.color_space,
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                cell_per_block=self.cell_per_block,
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img
