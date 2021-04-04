# CSC413PROJECT

### What is YOLO?
Yolo stands for you only look once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object. Before we get out hands dirty with code, we must understand how YOLO works.

Yolo makes uses of only convolutional layers, making it a fully convolutional network. It has 75 convolutional layers, with skip connections and upsampling layers. Norm form of pooling is used, and a convolutional layer with strid 2 is used to downsample the feature maps. This helps in preventing loss of low-level features often attributed to pooling.

Now, the first thing to notice is our output is a feature map. The size of the prediction map is exactly the size of the size of the feature map before it. The way you interpret this prediction map is that each cell can predict a fixed number of bounding boxes.

Depth-wise, we have (B x (5 + C)) entries in the feature map. B represents the number of bounding boxes each cell can predict. According to the paper, each of these B bounding boxes have 5 + C attributes, which describe the center coordinates, the dimensions, the objectness score and C class confidences for each bounding box.

You expect each cell of the feature map to predict an object through one of it's bounding boxes if the center of the object falls in the receptive field of that cell.

This has to do with how YOLO is trained, where only one bounding box is responsible for detecting any given object. First, we must ascertain which of the cells this bounding box belongs to.

To do that, we divide the input image into a grid of dimension equal to that of the final feature map.

Let us consider an example below, where the input image is 416 x x416, and stride of the network is 32. As pointed earlier, the dimensions of the feature map will be 13 x 13. We then divide the input image into 13 x 13 cells.

### Anchor Boxes
It might make sense to predict the width and the height of the bounding box, but in practice, that leads to unstable gradients during training. Instead, most of the modern object detectors predict log-space transforms, or simply offsets to pre-defined default bounding boxes called anchors.

The bounding box responsible for detecting the dog will be the one whose anchor has the highest IoU (intersection over union loss) with the ground truth box.
