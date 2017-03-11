## Vehicle Detection Project
---


[//]: # (Image References)
[image1]: ./report_pics/all_hot_windows1.png
[image2]: ./report_pics/all_hot_windows2.png
[image3]: ./report_pics/all_hot_windows3.png
[image4]: ./report_pics/all_hot_windows4.png
[image5]: ./report_pics/24_hot_windows_combined.png
[image6]: ./report_pics/24_heatmap.png
[image7]: ./report_pics/24_heatmap_combined.png
[image8]: ./report_pics/24_labels.png
[image9]: ./report_pics/24_final_detection.png
[image10]: ./report_pics/244_hot_windows_combined.png
[image11]: ./report_pics/103_hot_windows_combined.png

[video1]: ./project_video.mp4




### Building a classifier based on Histogram of Oriented Gradients (HOG) and other features

The code for this step is contained in the python script `build_classfier.py`. The parameters of the classifier can be set in lines 21-30. The function to extract features from an image is defined in lines 33-80 in the same file.
I started by reading in all the `vehicle` and `non-vehicle` images provided in the project repository.

[comment]: <> (Here is an example of one of each of the `vehicle` and `non-vehicle` classes:)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I also tried various combinations using other features discussed in the lectures such as `color histogram` and `spatial binning`. After some trial and error, I arrived at the following parameter settings:

* YUV color space
* HOG features for all color channels using:
    * 9 HOG orientations
    * 8 pixels per cell
    * 2 cells per block
* spatial binning for all channels with 16x16 bins
* color histogram for all channels with 16 bins

I tried to increase the number of pixels per cell to increase the speed of the whole pipeline.
However, the test accuracy was significantly lower, when I increased the number of pixels per cell.
With the above described parameter choice I trained a linear SVM after scaling the features using
a `StandardScaler()` and randomly splitting the data set into a test and training dataset.
I achieved a test accuracy of 99.13%. I then saved the SVC, the scaler, and all parameters into a `.pickle`file so that
it can be reused without having to retrain the classifier. The code for training the classifier can be found in lines 104-117 in the file `build_classfier.py`.

### Sliding Window Search
I decided to implement a multi-scale sliding window search based. To increase the processing speed of the pipeline I used the `hog_subsampling` function provided in the lecture as a starting point. This function allowed me to compute the HOG features all at once for each window "scale". I found that a four-scale approach with window sizes 64x64, 96x96, 128x128, and 196x196 provides nice results. Again to improve efficiency, I excluded the upper half of the frame from the sliding window search, starting the search at `y=400`. In addition, search using the smaller windows is focused on the middle part of the image, since the cars there will appear smaller. Cars in the lower part of the image will appear larger since they are closer. Therefore, the search using larger windows is focused on the lower part of the frame. The multi-scale sliding window search is implemented in the `CarDetector` class in the `CarDetector.py` file in the lines 32-37. The function doing the actual search can be found in the lines
91-151. During the sliding window search, the windows are usually advanced two cells at a time. However, I found that the algorithm performed better if I reduced this to one cell per step for the 96x96 window size an thereby increased the overlap.  


![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

 The combination of the SVC with the four-scale sliding window search is pretty reliable in detecting cars in a video frame and false negatives are rare. False positives occur more frequently. The images below show the detected "hot" windows for two frames from the project video.   


![alt text][image10]
![alt text][image11]
---

### Video Implementation
For the video processing pipeline, I focused on making the car detection more reliable. In particular I tried to reduce false positives using a heatmap averaged across multiple frames. Starting with the positive detections from the multi-scale sliding window search for an individual frame, I created a heatmap and then thresholded that map to identify vehicle positions. Then I combined this heatmap with the heatmaps from the four previous frames and then thresholded this heatmap again.

The motivation behind this is twofold. First, this technique reduces false positives. Second, it stabilizes the detection of the car. If a car is not detected in an individual frame, but has been correctly detected in the frames before, a window surrounding the car can still be produced using the heatmaps from previous frames.

Using these heatmaps averaged over five frames, I applied `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The series of images below
show how my pipeline works.

Starting with the positive detections from the sliding window search


![alt text][image5]

Calculated heatmap based on positive detections

![alt text][image6]

Averaged heatmap using the previous four frames

![alt text][image7]
The result of `scipy.ndimage.measurements.label()` to identify individual cars

![alt text][image8]

And the final bounding boxes surrounding the cars.

![alt text][image9]

The complete video can be found here.

Here's a [link to my video result](./output_final.mp4)

---

### Discussion
At first I was surprised by the high number of false positives after performing the sliding window search, given that the test accuracy of the svc was above 99%. Then I realized that
this behavior comes from the fact that I am essentially doing more than a hundred image classifications per frame, most which do not contain cars, which explains the relatively high number of false positives. Using thresholded heatmaps averaged over multiple frames reduces the amount of false positives drastically. However, there are still some false positives that are not filtered out by this approach, yet. Normally, I would start tweaking the algorithm further to get rid of these as well. Unfortunately, the end of term 1 is so close that I have no more time to do this.

One idea to improve the performance would be to use a second, different type of classifier in addition to the svc, e.g., a random forest or a deep neural net. One could use this second classifier to double check the detections produced by the first one. Or let them work in parallel and then combine the results.

Another weakness of my current implementation is the low processing speed. Processing the 50 second project video takes roughly 20 minutes on my MacBook Pro. One could probably try to speed up the multi-scale sliding window search and also try to use fewer scales. In addition it would make sense to avoid the sliding window search in every frame. Instead, one could try to search only in the vicinity of cars detected in the previous frames and in areas of the frame where new cars could enter, i.e., at the sides and the horizon. It would probably also make sense to include some kind of model for a car to predict its position from its location in previous frames.


The pipeline as it is implemented now has also several other limitations, e.g.:
  * It will be difficult to detect individual cars if the traffic increases
  * It will probably not work as well in different lighting conditions, e.g., at night
  * Since the classifier is only trained on normal cars, the pipeline will have problems with other types of vehicles such as truck, motorcycles, or busses.
