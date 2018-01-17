
# Finding Lane Lines on the Road

## Introduction:

Running the code using Python3 in Jupyter Notebook

Steps are:

1. Pre-process image using grayscale and gaussian blur
2. Apply canny edge detection to the image
3. Apply masking region to the image
4. Apply Hough transform to the image and extrapolate the lines found in the hough transform to construct the left and              right lane lines
5. Add the extrapolated lines to the input image

## Shortcomings:

* Region of interest should be selected with care.



## Important:

* All the output of test images are in test_images_output folder.
* All the output of test videos are in test_videos_output folder.
* Image output after each step is in examples folder.
