# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
import math

    
def initialization(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        

class Temperature_Scaling(nn.Module):
    def __init__(self, num_classes=None, image_channels=None):
        super(Temperature_Scaling, self).__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temp.data.fill_(1)

    def forward(self, logits, image=None, label=None):
        return logits / self.temp 


class Vector_Scaling(nn.Module):
    def __init__(self, num_classes, image_channels=None):
        super(Vector_Scaling, self).__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def weights_init(self):
        self.vector_parameters.data.fill_(1)
        self.vector_offset.data.fill_(0)

    def forward(self, logits, image=None, label=None):
        return (self.vector_parameters * logits) + self.vector_offset
        

class Dirichlet_Scaling(nn.Module):
    def __init__(self, num_classes, image_channels=None, eps=1e-10):
        super(Dirichlet_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_classes, num_classes)
        self.eps = eps

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits, image=None, label=None):
        probs = torch.softmax(logits, dim=1)
        ln_probs = torch.log(probs + self.eps)
        # Move channel dim to the back (for broadcasting)
        ln_probs = ln_probs.permute(0,2,3,1).contiguous()
        ds_probs = self.dirichlet_linear(ln_probs)
        ds_probs = ds_probs.permute(0,3,1,2).contiguous()
        # Return scaled log probabilities
        return ds_probs

        
class LTS(nn.Module):
    def __init__(self, num_classes, image_channels, sigma=1e-8):
        super(LTS, self).__init__()
        self.num_classes = num_classes

        self.temperature_level_2_conv1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.temperature_level_2_param1 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.temperature_level_2_conv_img = nn.Conv2d(image_channels, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_classes, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

        self.sigma = sigma

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

    def forward(self, logits, image, label=None):
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
        temperature = F.relu(temperature + torch.ones(1).cuda()) + self.sigma 
        temperature = temperature.repeat(1, self.num_classes, 1, 1)
        return logits / temperature


class Selective_Scaling(nn.Module):
    def __init__(self, num_classes, image_channels=None):
        super(Selective_Scaling, self).__init__()
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

    def forward(self, logits, images=None, labels=None):
        logits = logits.permute(0,2,3,1)   
        probs = torch.softmax(logits)
        ln_probs = torch.log(probs + 1e-16)

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
        
        return  tf_positive.permute(0,3,1,2)
        
        