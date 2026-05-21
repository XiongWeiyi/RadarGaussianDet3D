import torch
from mmdet3d.models import LOSSES
from torch import nn


@LOSSES.register_module()
class KLDivergenceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, scale_factors=None):
        super(KLDivergenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.scale_factors = scale_factors
        self.eps = 1e-8

    def forward(self, bbox_pred, bbox_gt, task_id=0):
        if len(bbox_pred) == 0:
            return torch.tensor(0., device=bbox_pred.device)

        sigma2_pred_z = ((bbox_pred[:, 5] / 2) ** 2).clamp(min=self.eps)
        sigma2_gt_z = (bbox_gt[:, 5] / 2) ** 2

        diff_z = bbox_pred[:, 2] - bbox_gt[:, 2]
        term_1 = (diff_z ** 2) / sigma2_gt_z
        term_2 = sigma2_pred_z / sigma2_gt_z
        term_3 = (sigma2_gt_z / sigma2_pred_z).log()

        s2_xy_pred = (bbox_pred[:, 3:5] / 2) ** 2
        s2_xy_gt = (bbox_gt[:, 3:5] / 2) ** 2
        rot_xy_pred = torch.stack([bbox_pred[:, 7], -bbox_pred[:, 6], bbox_pred[:, 6], bbox_pred[:, 7]], dim=-1).reshape(-1, 2, 2)
        rot_xy_gt = torch.stack([bbox_gt[:, 7], -bbox_gt[:, 6], bbox_gt[:, 6], bbox_gt[:, 7]], dim=-1).reshape(-1, 2, 2)
        cov_xy_pred = rot_xy_pred @ torch.diag_embed(s2_xy_pred) @ rot_xy_pred.permute(0, 2, 1)
        cov_xy_gt = rot_xy_gt @ torch.diag_embed(s2_xy_gt) @ rot_xy_gt.permute(0, 2, 1)

        cov_xy_pred_det = s2_xy_pred.prod(dim=-1).clamp(min=self.eps)
        cov_xy_gt_det = s2_xy_gt.prod(dim=-1)

        cov_xy_gt_inv = torch.stack([cov_xy_gt[:, 1, 1], -cov_xy_gt[:, 0, 1], -cov_xy_gt[:, 1, 0], cov_xy_gt[:, 0, 0]],
                                    dim=-1).reshape(-1, 2, 2) / cov_xy_gt_det[:, None, None]

        diff_xy = bbox_pred[:, :2] - bbox_gt[:, :2]
        term_1 = term_1 + (diff_xy[:, None, :] @ cov_xy_gt_inv @ diff_xy[:, :, None]).flatten()
        term_2 = term_2 + torch.einsum('bij,bji->b', cov_xy_gt_inv, cov_xy_pred)
        term_3 = term_3 + (cov_xy_gt_det / cov_xy_pred_det).log()

        if self.scale_factors is not None:
            if isinstance(self.scale_factors, list):
                term_1 = term_1 * self.scale_factors[task_id]
            else:
                term_1 = term_1 * self.scale_factors

        loss = 0.5 * (term_1 + term_2 + term_3 - 3.0).mean()

        if isinstance(self.loss_weight, list):
            return self.loss_weight[task_id] * loss
        else:
            return self.loss_weight * loss