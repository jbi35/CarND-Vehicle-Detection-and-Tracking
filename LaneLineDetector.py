from ImageProcessor import ImageProcessor
from LaneLine       import LaneLine
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
from moviepy.editor import VideoFileClip

# Define a class to handle lane line detection
class LaneLineDetector:
    def __init__(self,debug):
        self.my_image_processor = ImageProcessor()

        self.left_lane_line = LaneLine()

        self.right_lane_line = LaneLine()

        self.debug=debug

    def apply_pipeline(self,img):

        # undistort image
        undistorted_img = self.my_image_processor.undistort_image(img)

        # apply thresholds
        thresholded_img = self.my_image_processor.compute_binary_thresholded_image(undistorted_img)

        # warp into brids eye view
        transformed_img = self.my_image_processor.apply_perspective_transform(thresholded_img)

        # perform lane line detection
        lane_img, left_fitx, right_fitx, ploty, position, left_curverad, right_curverad = self.get_lane_lines(transformed_img)

        # draw lane line detection on distorted image
        final_image = self.my_image_processor.draw_lanes_on_road(undistorted_img,lane_img,ploty,left_fitx,right_fitx)
        final_image = self.my_image_processor.add_curve_radius_and_car_pos_to_images(final_image,position,left_curverad,right_curverad)

        ## in debug mode show all images
        if debug:
            plt.imshow(undistorted_img)
            plt.show()
            plt.imshow(cv2.cvtColor(thresholded_img*255, cv2.COLOR_GRAY2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(transformed_img*255, cv2.COLOR_GRAY2RGB))
            plt.show()
            plt.imshow(lane_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
            plt.imshow(final_image)
            plt.show()
        #and we are done
        return final_image

    def process_image(self, undistorted_img):
        #undistorted_img = self.my_image_processor.undistort_image(img)
        thresholded_img = self.my_image_processor.compute_binary_thresholded_image(undistorted_img)
        #thresholded_img = self.my_image_processor.apply_all_thresholds(undistorted_img)
        transformed_img = self.my_image_processor.apply_perspective_transform(thresholded_img)
        return transformed_img

    def get_lane_lines(self,transformed_img):

        peaks, histogram = self.my_image_processor.compute_histogram_with_peaks(transformed_img)
        leftx_base = peaks[0]
        rightx_base = peaks[1]

        # advanced lane line detection
        lane_binary_left =  self.left_lane_line.fit_lane_line(transformed_img,leftx_base)
        lane_binary_right =  self.right_lane_line.fit_lane_line(transformed_img,rightx_base)

        #lane_binary_left =  self.left_lane_line.fit_lane_line_simple(transformed_img,leftx_base)
        #lane_binary_right =  self.right_lane_line.fit_lane_line_simple(transformed_img,rightx_base)

        left_fitx = self.left_lane_line.get_allx()
        right_fitx = self.right_lane_line.get_allx()
        ploty = self.left_lane_line.get_ally()

        combined_binary = np.zeros_like(lane_binary_right)
        out_img = np.dstack((lane_binary_right, lane_binary_left, combined_binary))*255
        new_curverad_l = self.left_lane_line.get_curvature()
        new_curverad_r = self.right_lane_line.get_curvature()
        lane_dist_right = self.right_lane_line.get_lane_position()
        lane_dist_left = self.left_lane_line.get_lane_position()
        w_lane = lane_dist_right - lane_dist_left
        position = w_lane/2 - lane_dist_right

        return out_img, left_fitx, right_fitx, ploty, position, new_curverad_l, new_curverad_r

    def process_test_images(self):
        #for i,img_name in enumerate(("camera_cal/calibration3.jpg", "test_images/straight_lines1.jpg")):
        for img_name in glob.glob("test_images/*.jpg"):
            print(img_name)
            img = mpimg.imread(img_name)

            # undistort image
            undistorted_img = self.my_image_processor.undistort_image(img)

            #transformed_img = self.my_image_processor.apply_perspective_transform(undistorted_img)

            # apply thresholds
            thresholded_img = self.my_image_processor.compute_binary_thresholded_image(undistorted_img)

            # warp into brids eye view
            transformed_img = self.my_image_processor.apply_perspective_transform(thresholded_img)

            # perform lane line detection
            lane_img, left_fitx, right_fitx, ploty, position, left_curverad, right_curverad = self.get_lane_lines(transformed_img)

            # draw lane line detection on distorted image
            final_image = self.my_image_processor.draw_lanes_on_road(undistorted_img,lane_img,ploty,left_fitx,right_fitx)
            final_image = self.my_image_processor.add_curve_radius_and_car_pos_to_images(final_image,position,left_curverad,right_curverad)


            mpimg.imsave('undistorted_images/'+img_name, undistorted_img)

            mpimg.imsave('thresholded_images/'+img_name, cv2.cvtColor(thresholded_img*255, cv2.COLOR_GRAY2RGB))

            mpimg.imsave('transformed_images/'+img_name,transformed_img)
            #mpimg.imsave('transformed_images/'+img_name, cv2.cvtColor(transformed_img*255, cv2.COLOR_GRAY2RGB))

            mpimg.imsave('final_images/'+img_name, final_image)

            plt.imshow(lane_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            ## save those manually
            plt.show()



    def process_test_video(self,input_file,output_file,start=0.0,end=2.0):
        video = VideoFileClip(input_file)
        #video = video.subclip(t_start=start, t_end=end)
        processed_video = video.fl_image(self.apply_pipeline)
        processed_video.write_videofile(output_file,audio=False)


    def undistort_calibration_images_for_writeup(self):
        for img_name in glob.glob("camera_cal/*.jpg"):
            print(img_name)
            img = mpimg.imread(img_name)
            undistorted_img = self.my_image_processor.undistort_image(img)
            mpimg.imsave('undistorted_images/'+img_name, undistorted_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced lane line detection")
    parser.add_argument('--input', type=str, default='project_video.mp4', help='input video')
    parser.add_argument('--output', default='output.mp4', type=str, help='output video')
    parser.add_argument('--debug', type=str, default='no', help='debug mode yes/no')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    if args.debug == 'yes':
        debug = True
    elif args.debug == 'no':
        debug = False
    else:
        print('Warning input flag not set correctly not showing debug information')
        debug = False

    print('Input file: {}'.format(input_file))
    print('Output file: {}'.format(output_file))
    print('Debug mode: {}'.format(debug))

    my_lanes_line_detector=LaneLineDetector(debug)

    my_lanes_line_detector.process_test_video(input_file,output_file)
    #my_lanes_line_detector.undistort_calibration_images_for_writeup()
    #my_lanes_line_detector.process_test_images()
