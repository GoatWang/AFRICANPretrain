#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# class SoftTargetCrossEntropy(nn.Module):
#     """
#     Cross entropy loss with soft target.
#     """

#     def __init__(self, reduction="mean"):
#         """
#         Args:
#             reduction (str): specifies reduction to apply to the output. It can be
#                 "mean" (default) or "none".
#         """
#         super(SoftTargetCrossEntropy, self).__init__()
#         self.reduction = reduction

#     def forward(self, x, y):
#         loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "none":
#             return loss
#         else:
#             raise NotImplementedError

class BCELoss(nn.Module):
    '''
    Function: BCELoss
    Params:
        predictions: input->(batch_size, 1004)
        targets: target->(batch_size, 1004)
    Return:
        bceloss
    '''

    def __init__(self, logits=True, reduction="mean"):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)

        return BCE_loss

class FocalLoss(nn.Module):
    '''
    Function: FocalLoss
    Params:
        alpha: scale factor, default = 1
        gamma: exponential factor, default = 0
    Return:
        focalloss
    https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
    Original: https://github.com/facebookresearch/Detectron
    '''

    def __init__(self, logits=True, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = 1 
        self.gamma = 5 # adjusted from 0 to 5
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss

_LOSSES = {
    "BCE": BCELoss,
    "FOCAL": FocalLoss,
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

if __name__ == "__main__":
    def test_focal_loss():
        focal_loss = FocalLoss([], 'cpu')

        # Scenario 1: low loss case
        # We take logits (model's raw outputs) to be very close to targets
        logits1 = torch.tensor([5.0, -5.0, 5.0, -5.0])  # assuming these as model's raw outputs
        targets1 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # ground truth labels
        loss1 = focal_loss(logits1, targets1)

        # Scenario 2: high loss case
        # We take logits to be very far from targets
        logits2 = torch.tensor([-5.0, 5.0, -5.0, 5.0])  # assuming these as model's raw outputs
        targets2 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # ground truth labels
        loss2 = focal_loss(logits2, targets2)

        print('Loss in low loss case:', loss1.item())
        print('Loss in high loss case:', loss2.item())

        assert loss2 > loss1, "Focal loss in the high loss case should be larger than in the low loss case"

    test_focal_loss()







    # import numpy as np
    # import pandas as pd
    # df = pd.read_excel("/Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/annotation/df_action.xlsx")
    # class_frequency = list(map(float, df["count"].tolist()))

    # loss_result = []
    # for loss_name in _LOSSES.keys():
    #     loss_func = get_loss_func(loss_name)(class_frequency, 'cpu', logits=True)        

    #     ZERO_LOGINVERSE = -13.815509557935018
    #     ONE_LOGINVERSE = 13.815509557935018
    #     EPS = 10**-6

    #     input_zeros = torch.ones((10, 140)) * ZERO_LOGINVERSE
    #     target_zeros = torch.zeros((10, 140))
    #     loss_zeros = loss_func(input_zeros, target_zeros)

    #     input_hfp = input_zeros.clone()
    #     idxs_tn = torch.where((input_hfp < (ZERO_LOGINVERSE + EPS)) & (target_zeros < 0.5))
    #     idxs_tn_cand_idxs = np.random.choice(range(len(idxs_tn[0])), int(0.3*len(idxs_tn[0])), replace=False)
    #     # idxs_tn_cand_idxs.sort()
    #     idxs_tn_cands = [idx_tn[idxs_tn_cand_idxs] for idx_tn in idxs_tn]
    #     input_hfp[idxs_tn_cands] = ONE_LOGINVERSE * 0.5
    #     loss_hfp_half = loss_func(input_hfp, target_zeros)

    #     input_hfp = input_zeros.clone()
    #     idxs_tn = torch.where((input_hfp < (ZERO_LOGINVERSE + EPS)) & (target_zeros < 0.5))
    #     idxs_tn_cand_idxs = np.random.choice(range(len(idxs_tn[0])), int(0.3*len(idxs_tn[0])), replace=False)
    #     # idxs_tn_cand_idxs.sort()
    #     idxs_tn_cands = [idx_tn[idxs_tn_cand_idxs] for idx_tn in idxs_tn]
    #     input_hfp[idxs_tn_cands] = ONE_LOGINVERSE
    #     loss_hfp = loss_func(input_hfp, target_zeros)



    #     input_ones = torch.ones((10, 140)) * ONE_LOGINVERSE
    #     target_ones = torch.ones((10, 140))
    #     loss_ones = loss_func(input_ones, target_ones)

    #     input_hfn = input_ones.clone()
    #     idxs_tp = torch.where((input_hfn > (ONE_LOGINVERSE - EPS)) & (target_ones > 0.5))
    #     idxs_tp_cand_idxs = np.random.choice(range(len(idxs_tp[0])), int(0.3*len(idxs_tp[0])), replace=False)
    #     # idxs_tp_cand_idxs.sort()
    #     idxs_tp_cands = [idx_tp[idxs_tp_cand_idxs] for idx_tp in idxs_tp]
    #     input_hfn[idxs_tp_cands] = ZERO_LOGINVERSE * 0.5
    #     loss_hfn_half = loss_func(input_hfn, target_ones)
    #     print(input_hfn)

    #     input_hfn = input_ones.clone()
    #     idxs_tp = torch.where((input_hfn > (ONE_LOGINVERSE - EPS)) & (target_ones > 0.5))
    #     idxs_tp_cand_idxs = np.random.choice(range(len(idxs_tp[0])), int(0.3*len(idxs_tp[0])), replace=False)
    #     # idxs_tp_cand_idxs.sort()
    #     idxs_tp_cands = [idx_tp[idxs_tp_cand_idxs] for idx_tp in idxs_tp]
    #     input_hfn[idxs_tp_cands] = ZERO_LOGINVERSE
    #     loss_hfn = loss_func(input_hfn, target_ones)
    #     print(input_hfn)

    #     loss_result.append({
    #         'loss_zeros': loss_zeros.item(),
    #         'loss_hfp_half': loss_hfp_half.item(),
    #         'loss_hfp': loss_hfp.item(),
    #         'loss_ones': loss_ones.item(),
    #         'loss_hfn_half': loss_hfn_half.item(),
    #         'loss_hfn': loss_hfn.item()
    #     })
    
    # # columns = 'loss_ori', 'loss_hfp', 'loss_hfn'
    # df_loss = pd.DataFrame(loss_result)# , columns=columns
    # df_loss.index = _LOSSES.keys()
    # print(df_loss)