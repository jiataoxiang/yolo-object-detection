"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from util import IoU


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambdaNoobj = 0.5
        self.lambdaCoord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # print(predictions.shape)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        IoUB1 = IoU(predictions[..., self.C + 1 : self.C + 5], target[..., self.C + 1 : self.C + 5])
        IoUB2 = IoU(predictions[..., self.C + 6 : self.C + 10], target[..., self.C + 1 : self.C + 5])
        IoUs = torch.cat([IoUB1.unsqueeze(0), IoUB2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        IoUMaxes, bestbox = torch.max(IoUs, dim=0)
        # print(bestbox.shape)
        existsBox = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        boxPredictions = existsBox * (
            (
                bestbox * predictions[..., self.C + 6 : self.C + 10]
                + (1 - bestbox) * predictions[..., self.C + 1:self.C + 5]
            )
        )

        boxTargets = existsBox * target[..., self.C + 1:self.C + 5]

        # Take sqrt of width, height of boxes to ensure that
        boxPredictions[..., 2:4] = torch.sign(boxPredictions[..., 2:4]) * torch.sqrt(
            torch.abs(boxPredictions[..., 2:4] + 1e-6)
        )
        boxTargets[..., 2:4] = torch.sqrt(boxTargets[..., 2:4])

        boxLoss = self.mse(
            torch.flatten(boxPredictions, end_dim=-2),
            torch.flatten(boxTargets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # predBox is the confidence score for the bbox with highest IoU
        predBox = (
            bestbox * predictions[..., self.C + 5:self.C + 6] + (1 - bestbox) * predictions[..., self.C:self.C + 1]
        )
        objectLoss = self.mse(
            torch.flatten(existsBox * predBox),
            torch.flatten(existsBox * target[..., self.C:self.C + 1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        NoObjectLoss = self.mse(
            torch.flatten((1 - existsBox) * predictions[..., self.C:self.C + 1], start_dim=1),
            torch.flatten((1 - existsBox) * target[..., self.C:self.C + 1], start_dim=1),
        )

        NoObjectLoss += self.mse(
            torch.flatten((1 - existsBox) * predictions[..., self.C + 5:self.C + 6], start_dim=1),
            torch.flatten((1 - existsBox) * target[..., self.C:self.C + 1], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        classLoss = self.mse(
            torch.flatten(existsBox * predictions[..., :self.C], end_dim=-2,),
            torch.flatten(existsBox * target[..., :self.C], end_dim=-2,),
        )

        loss = (
            self.lambdaCoord * boxLoss  # first two rows in paper
            + objectLoss  # third row in paper
            + self.lambdaNoobj * NoObjectLoss  # forth row
            + classLoss  # fifth row
        )

        return loss