import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np


class Temperature_Scaling(nn.Module):
    def __init__(self):
        super(Temperature_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temperature_single.data.fill_(1)

    def forward(self, logits):
        temperature = self.temperature_single.expand(logits.size()).cuda()
        return logits / temperature


class Vector_Scaling(nn.Module):
    def __init__(self, num_classes):
        super(Vector_Scaling, self).__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def weights_init(self):
        self.vector_offset.data.fill_(0)
        self.vector_parameters.data.fill_(1)

    def forward(self, logits):
        return logits * self.vector_parameters.cuda() + self.vector_offset.cuda()
        

class Stochastic_Spatial_Scaling(nn.Module):
    def __init__(self, num_classes):
        super(Stochastic_Spatial_Scaling, self).__init__()

        conv_fn = nn.Conv2d
        self.rank = 10
        self.num_classes = num_classes 
        self.epsilon = 1e-5
        self.diagonal = False  # whether to use only the diagonal (independent normals)
        self.conv_logits = conv_fn(num_classes, num_classes, kernel_size=(1, ) * 2)

    def weights_init(self):
        initialization(self.conv_logits)
        
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean
                
    def forward(self, logits):

        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]


        mean = self.conv_logits(logits)
        cov_diag = (mean*1e-5).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))                     

        base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        distribution = ReshapedDistribution(base_distribution, event_shape)

        num_samples=2
        samples = distribution.rsample((num_samples // 2,)).cpu()
        mean = distribution.mean.unsqueeze(0).cpu()
        samples = samples - mean
        logit_samples = torch.cat([samples, -samples]) + mean
        logit_mean = logit_samples.mean(dim=0).cuda()

        return logit_mean
        

class Dirichlet_Scaling(nn.Module):
    def __init__(self, num_classes):
        super(Dirichlet_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits):
        logits = logits.permute(0,2,3,1)
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        ln_probs = torch.log(probs+1e-10)

        return self.dirichlet_linear(ln_probs).permute(0,3,1,2)   

        
class Meta_Scaling(nn.Module):
    def __init__(self, num_classes):
        super(Meta_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))
        self.num_classes = num_classes
        self.alpha = 0.05

    def weights_init(self):
        self.temperature_single.data.fill_(1)
        
    def forward(self, logits, gt, threshold):

        logits = logits.permute(0,2,3,1).view(-1, self.num_classes)
        gt = gt.view(-1)
    
        if self.training:
            neg_ind = torch.argmax(logits, axis=1) == gt
            
            xs_pos, ys_pos = logits[~neg_ind], gt[~neg_ind]
            xs_neg, ys_neg = logits[neg_ind], gt[neg_ind]
            
            start = np.random.randint(int(xs_neg.shape[0]*1/3))+1
            x2 = torch.cat((xs_pos, xs_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            y2 = torch.cat((ys_pos, ys_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            cal_logits, cal_gt = x2[cond_ind], y2[cond_ind]
        
            temperature = self.temperature_single.expand(cal_logits.size())
            cal_logits = cal_logits / temperature
            
        else:
            x2 = logits
            y2 = gt
        
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            scaled_logits, scaled_gt = x2[cond_ind], y2[cond_ind]
            inference_logits, inference_gt = x2[~cond_ind], y2[~cond_ind]
        
            temperature = self.temperature_single.expand(scaled_logits.size())
            scaled_logits = scaled_logits / temperature

            inference_logits = torch.ones_like(inference_logits)
            
            cal_logits = torch.cat((scaled_logits, inference_logits), 0)
            cal_gt = torch.cat((scaled_gt, inference_gt), 0)

        return cal_logits, cal_gt


class LTS_CamVid_With_Image(nn.Module):
    def __init__(self, num_classes):
        super(LTS_CamVid_With_Image, self).__init__()
        self.num_classes = num_classes
        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_num_class = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_num_class * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature


class Binary_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Binary_Classifier, self).__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)
        self.binary_linear = nn.Linear(num_classes, 2)
        
        self.bn0 = nn.BatchNorm2d(num_classes)
        self.linear_1 = nn.Linear(num_classes, num_classes*2)
        self.bn1 = nn.BatchNorm2d(num_classes*2)
        self.linear_2 = nn.Linear(num_classes*2, num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)

        self.relu = nn.ReLU()        

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits, gt):
        logits = logits.permute(0,2,3,1)   
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)

        ln_probs = torch.log(probs+1e-16)

        out = self.dirichlet_linear(ln_probs)
        out = self.bn0(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_1(out.permute(0,2,3,1))
        out = self.bn1(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_2(out.permute(0,2,3,1))
        out = self.bn2(out.permute(0,3,1,2))
        out = self.relu(out)       

        tf_positive = self.binary_linear(out.permute(0,2,3,1))
        _, pred = torch.max(probs, dim=-1)
        
        mask = pred == gt
        
        return  tf_positive.permute(0,3,1,2), mask.long()
        
        