import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

num_class = 2

def IoU(predictions, targetLabels):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    
    # preds_topLeft_x = (predictions[..., 0] - predictions[..., 2] / 2).unsqueeze(-1)
    # preds_topLeft_y = (predictions[..., 1] - predictions[..., 3] / 2).unsqueeze(-1)
    # preds_bottomRight_x = (predictions[..., 0] + predictions[..., 2] / 2).unsqueeze(-1)
    # preds_bottomRight_y = (predictions[..., 1] - predictions[..., 3] / 2).unsqueeze(-1)
    
    # target_topLeft_x = (targetLabels[..., 0] - targetLabels[..., 2] / 2).unsqueeze(-1)
    # target_topLeft_y = (targetLabels[..., 1] - targetLabels[..., 3] / 2).unsqueeze(-1)
    # target_bottomRight_x = (targetLabels[..., 0] + targetLabels[..., 2] / 2).unsqueeze(-1)
    # target_bottomRight_y = (targetLabels[..., 1] - targetLabels[..., 3] / 2).unsqueeze(-1)
    preds_topLeft_x = (predictions[..., 0] - predictions[..., 2] / 2).unsqueeze(-1)
    preds_topLeft_y = (predictions[..., 1] - predictions[..., 3] / 2).unsqueeze(-1)
    preds_bottomRight_x = (predictions[..., 0] + predictions[..., 2] / 2).unsqueeze(-1)
    preds_bottomRight_y = (predictions[..., 1] + predictions[..., 3] / 2).unsqueeze(-1)
    target_topLeft_x = (targetLabels[..., 0] - targetLabels[..., 2] / 2).unsqueeze(-1)
    target_topLeft_y = (targetLabels[..., 1] - targetLabels[..., 3] / 2).unsqueeze(-1)
    target_bottomRight_x = (targetLabels[..., 0] + targetLabels[..., 2] / 2).unsqueeze(-1)
    target_bottomRight_y = (targetLabels[..., 1] + targetLabels[..., 3] / 2).unsqueeze(-1)

	# variables used to calculate intersection
    max_topLeft_x = torch.max(preds_topLeft_x, target_topLeft_x)
    max_TopLeft_y = torch.max(preds_topLeft_y, target_topLeft_y)
    min_BottomRight_x = torch.min(preds_bottomRight_x, target_bottomRight_x)
    min_BottomRight_y = torch.min(preds_bottomRight_y, target_bottomRight_y)

	# Set min value to 0 if they do not overlap
    intersection = (min_BottomRight_x - max_topLeft_x).clamp(0) * (min_BottomRight_y - max_TopLeft_y).clamp(0)

    predictionArea = abs((preds_topLeft_x - preds_bottomRight_x) * (preds_topLeft_y - preds_bottomRight_y))
    targetArea = abs((target_topLeft_x - target_bottomRight_x) * (target_topLeft_y - target_bottomRight_y))

    return intersection / (predictionArea + targetArea - intersection + 1e-6)


def EliminateLowerProbability(boundingBoxes, threshold):
	"""
		Helper for non max suppression
		Eliminate bounding boxes if boundingBoxes[compareIndex] value is less than threshold
	"""
	index = 0
	while index < len(boundingBoxes):
		if boundingBoxes[index][1] < threshold:
			boundingBoxes.pop(index)
		index += 1
	return boundingBoxes


def EliminateDuplicateBox(boundingBoxes, threshold):
	"""
		Helper for non max suppression
		Eliminates the boxes if previous class have the same class and IoU > 0.5 with current prediction
	"""
	res = []
	while boundingBoxes:
		curBox = boundingBoxes.pop(0)
		# if other box have same class pred as cur Box and IoU with curBox > threshold, remove them
		index = 0
		while index < len(boundingBoxes):
			box = boundingBoxes[index]
			IoU_score = IoU(torch.tensor(curBox[2:]), torch.tensor(box[2:]))
			if box[0] == curBox[0] and IoU_score > threshold:
				boundingBoxes.pop(index)
			index += 1
		res.append(curBox)
	return res

"""
	Possible implementation according to https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
	1. Sort the predictions by the confidence scores.
	2. Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
	3. Repeat step 2 until all predictions are checked.
"""
def nonMaxSuppression(boundingBoxes, IoUThreshold, Probabilitythreshold):
	"""
		Given bounding boxes, eliminates those with low probability given threshold
		Parameters:
			boundingBoxes (list): list of bounding boxes
			[class, probability, x, y, w, h], where (x, y) is the midpoint of the bounding boxes
			IoUThreshold: threshold to find probably correct boxes
			Probabilitythreshold: threshold to remove boxes
		Returns:
		list: boxes that have higher than Probabilitythreshold probability and higher than
		IoUThreshold IoU
	"""
	# For debug purpose
	assert type(boundingBoxes) == list

	# Eliminate boxes with probability lower than Probabilitythreshold
	boundingBoxes = EliminateLowerProbability(boundingBoxes, Probabilitythreshold)

	# Sort by confident score from high to low
	boundingBoxes = sorted(boundingBoxes, key=lambda x: x[1], reverse=True)

	# Eliminate boxes with IoU lower than IoUThreshold
	# Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.
	return EliminateDuplicateBox(boundingBoxes, IoUThreshold)


def MeanAveragePrecision(
    predictionBoxes, targetBoxes, IoUThreshold, numClass=2
):
    """
		Calculates the mean average precision, which is the ratio of true positive over all cases
		Parameters:
			predictionBoxes (list): list of all predictions bounding boxes
			specified as [train_idx, class, probability score, x, y, w, h]
			targetBoxes (list): list of all target boxes
			Same format as predictionBoxes
			IoUThreshold (float): threhold for correct predicted boxes
			numClass: number of classes
		Returns:
			float: mAP value across all classes
	"""
    # store precisions for each classes
    averagePrecisions = []
    # Handle numerical instability issue
    epsilon = 1e-6 

    # loop through for perticular class class_
    for class_ in range(numClass):
        # go through all detections and target, and find class_
        detections = [detection for detection in predictionBoxes if detection[1] == class_]
        groundTruth = [trueBox for trueBox in targetBoxes if trueBox[1] == class_]

        
        # find amount of bounding boxes for each training examples
        amountBoxes = Counter([gt[0] for gt in groundTruth])

        for key, val in amountBoxes.items():
            amountBoxes[key] = torch.zeros(val)

        # sort by box probability
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        totalTrueBoxes = len(groundTruth)

        # skip if non for this class
        if totalTrueBoxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # compare with the same training images
            groundTruthImage = [
                box for box in groundTruth if box[0] == detection[0]
            ]

            num_gts = len(groundTruthImage)
            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(groundTruthImage):
                iou = IoU(torch.tensor(detection[3:]), torch.tensor(gt[3:]))

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > IoUThreshold:
                if amountBoxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amountBoxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (totalTrueBoxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        averagePrecisions.append(torch.trapz(precisions, recalls))

    return sum(averagePrecisions) / len(averagePrecisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def getBoundingBoxes(loader, model, IoUThreshold, Probabilitythreshold, device="cuda"):
    """
		Get all prediction bounding boxes and ground truth bounding boxes
	"""
    predictionBoxes = []
    targetBoxes = []
	# set model to eval so that won't trigger back propagation
    model.eval()
    train_idx = 0

    for batch_index, (image, labels) in enumerate(loader):
        image = image.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(image)

        batch_size = image.shape[0]

        # convert label matrices to correct format
        targetBox = cellboxes_to_boxes(labels)
        predictionBox = cellboxes_to_boxes(predictions)

        for i in range(batch_size):
            # get boxes after non max suppression
            nms_boxes = nonMaxSuppression(predictionBox[i], IoUThreshold, Probabilitythreshold)

            for nms_box in nms_boxes:
                predictionBoxes.append([train_idx] + nms_box)

            for box in targetBox[i]:
                if box[1] > Probabilitythreshold:
                    targetBoxes.append([train_idx] + box)

            train_idx += 1
    # set model back to train
    model.train()
    return predictionBoxes, targetBoxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, -1)
    bboxes1 = predictions[..., num_class + 1:num_class + 5]
    bboxes2 = predictions[..., num_class + 6:num_class + 10]
    scores = torch.cat(
        (predictions[..., num_class].unsqueeze(0), predictions[..., num_class + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :num_class].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., num_class], predictions[..., num_class + 5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])