import numpy as np

# Define a class to receive the characteristics of each line detection
class LaneLine():
    def __init__(self):
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_dist = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # keep track of the number of frames we did not have a good fit
        self.num_frames_without_fit = 0
        # meters per pixel in y dimension
        self.ym_per_pix = 30/720
        # meters per pixel in x dimension
        self.xm_per_pix = 3.7/700
        # width of frame
        self.frame_width = 1280

    def fit_lane_line_simple(self,img,anchor):
        x, y, lane_inds, out_img = self.get_lane_lines_pixels_using_sliding_windows(img,anchor)
        self.current_fit = self.fit_polynomial(x,y)
        self.best_fit = self.current_fit
        # compute actual lane line pixels
        self.ally = np.linspace(0, img.shape[0]-1, img.shape[0] )
        self.allx= self.best_fit[0]*self.ally **2 + self.best_fit[1]*self.ally  + self.best_fit[2]
        self.compute_curvature_and_position()
        return out_img

    def fit_lane_line(self,img,anchor):

        if self.got_good_fit():
            # if we have good fit in previous frame
            x, y, lane_inds, out_img = self.get_lane_line_pixels_using_previous_fit(img)
            # TODO add basic sanity check
            if len(x) < 10:
                self.current_fit = None
            else:
                self.current_fit = self.fit_polynomial(x,y)
        else:
            x, y, lane_inds, out_img = self.get_lane_lines_pixels_using_sliding_windows(img,anchor)
            self.current_fit = self.fit_polynomial(x,y)

        self.update_fit()

        # compute actual lane line pixels
        self.ally = np.linspace(0, img.shape[0]-1, img.shape[0] )
        self.allx= self.best_fit[0]*self.ally **2 + self.best_fit[1]*self.ally  + self.best_fit[2]

        self.compute_curvature_and_position()
        return out_img

    def get_lane_lines_pixels_using_sliding_windows(self,binary_warped,anchor):
        """
            search for lane line pixels using sliding windows
        """
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = anchor
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        lane_binary = np.zeros_like(binary_warped)
        lane_binary[nonzeroy[lane_inds], nonzerox[lane_inds]] = 1

        # return pixels identified as lane lines
        return x, y, lane_inds, lane_binary

    def get_lane_line_pixels_using_previous_fit(self,binary_warped):
        """
            search for lane line pixels using result from previous fit
        """
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 25
        lane_inds = ((nonzerox > (self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + self.best_fit[2] - margin)) & (nonzerox < (self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + self.best_fit[2] + margin)))

        # Again, extract pixel positions and pixels identified as lane lines
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Create an image to draw on and an image to show the selection window
        lane_binary = np.zeros_like(binary_warped)
        lane_binary[nonzeroy[lane_inds], nonzerox[lane_inds]] = 1
        return x, y, lane_inds, lane_binary

    def got_good_fit(self):
        if self.best_fit != None and self.num_frames_without_fit < 5:
            return True
        else:
            return False

    def fit_polynomial(self,mx,my):
        pol = np.polyfit(my,mx,2)
        return pol

    def update_fit(self):
        if self.current_fit is None:
            self.num_frames_without_fit += 1
            return

        elif self.best_fit is None:
            self.best_fit = self.current_fit

        else:
            self.update_polynomial(self.current_fit)


    # update polynomial fit weighting previous fits
    def update_polynomial(self, p):
        a = 0.25
        b = 1.0 - a
        self.best_fit = a * p + b * self.best_fit

    def compute_curvature_and_position(self):
        y_eval = np.max(self.ally)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        # position w.r.t lane centre from
        distance_in_pixel = (self.best_fit[0]*y_eval **2 + self.best_fit[1]*y_eval + self.best_fit[2])  - self.frame_width / 2
        self.lane_dist = distance_in_pixel * self.xm_per_pix


    def get_curvature(self):
        return self.radius_of_curvature

    def get_lane_position(self):
        return self.lane_dist

    def get_allx(self):
        return self.allx

    def get_ally(self):
        return self.ally
