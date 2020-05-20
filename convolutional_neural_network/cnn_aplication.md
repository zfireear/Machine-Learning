# CNN Aplication

## Object Detection
Classification with localization(object localization)
$$L(\hat{y},y) = 
\begin{cases}
  (\hat{y_1} - y_1)^2 + \cdots + (\hat{y_m} - y_m)^2, &  y_1 = 1 \\
  (\hat{y_1} - y_1)^2, & y_1 = 0
\end{cases}
$$
output contains such as $b_x$,$b_y$,$b_H$,$b_W$ bounding box and class babel(1-n), eg:
$$y = \begin{bmatrix} P_c\\b_x \\ b_y \\ b_H \\ b_W \\ C_1 \\ C_2 \\C_3 \end{bmatrix}$$

## Landmark Detection
Adding a bunch of output unit to output the `x,y` coordinates of different landmark you want to recognize. The identity of landmark must be consistent across different images.

## Sliding Windows Detection Convolutional Implementation
Share a lot of computation in the rigions of the image that are commom, it combines all into one for computation. And convolutionally make all the predictions at the same time by one for it pass through this big ConvNet.

## YOLO Algorithm
It takes the midpoint of each of the goal object and it assigns the object to grid cell containing the midpoint.(It's also one single convolutional implementation)  
midpoint : $b_x$, $b_y$, $b_H$, $b_W$
It takes image classification and localization algothm to output more accurate bounding boxes.

### Intersection over Union(IoU)
It computes the intersection over union of predicted and ground truth bounding boxes.
$$= \dfrac{size\quad of\quad intersection\quad area}{size\quad of\quad union\quad area}$$
"correct" if IoU $\geq$ 0.5
IoU can be used to evaluate whether or not you're object localization algorithm is accurate, IoU is a measure of the overlap between two bounding boxes. 