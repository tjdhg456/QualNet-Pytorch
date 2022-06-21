import torch
import torch.nn as nn
import torch.nn.functional as F
from metric.utils import pdist
import numpy as np

class RKdPoint(nn.Module):
    def __init__(self):
        super(RKdPoint, self).__init__()

    def forward(self, student, teacher):
        loss = F.smooth_l1_loss(student, teacher)
        return loss

class RKdAngle(nn.Module):
    def __init__(self):
        super(RKdAngle, self).__init__()

    def forward(self, student, teacher):
        with torch.no_grad():
            # For Memory
            device = teacher.device
            teacher = teacher.cpu()
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            td = td.to(device)

            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RKdDistance(nn.Module):
    def __init__(self):
        super(RKdDistance, self).__init__()

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

class RKdCenter(nn.Module):
    def __init__(self, center_path, num_classes, device):
        super(RKdCenter, self).__init__()
        self.center = torch.from_numpy(np.load(center_path)).float().to(device)
        self.rkd = RKdDistance()
        self.num_classes = num_classes

    def forward(self, student, teacher):
        batch_size = student.size(0)

        distmat_student = torch.pow(student, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.center, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_student.addmm_(1, -2, student, self.center.t())

        distmat_teacher = torch.pow(teacher, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.center, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat_teacher.addmm_(1, -2, teacher, self.center.t())
        loss = self.rkd(distmat_student, distmat_teacher)
        return loss

class distill_cri(nn.Module):
    def __init__(self, args, device):
        super(distill_cri, self).__init__()
        self.criterion_r1 = RKdPoint()
        self.criterion_r2 = RKdDistance()
        self.criterion_r3 = RKdAngle()

        self.param_r1 = 1.
        self.param_r2 = 0.02
        self.param_r3 = 0.01

    def forward(self, student, teacher):
        student, teacher = student.reshape(student.size(0), -1), teacher.reshape(teacher.size(0), -1)
        r1 = self.param_r1 * self.criterion_r1(student, teacher)
        r2 = self.param_r2 * self.criterion_r2(student, teacher)
        r3 = self.param_r3 * self.criterion_r3(student, teacher)
        loss_tot = (r1+r2+r3)
        return loss_tot


