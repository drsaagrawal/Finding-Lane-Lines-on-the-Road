
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

* It might fail in curve.
* It might fail in roads with improper lights.



## Improvement:

* Can be improved by implementing approach for dynamic region of interest.
* A approach can be implemented to take care of all road conditions.
