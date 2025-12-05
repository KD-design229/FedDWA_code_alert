# Several basic machine learning models
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
from torchvision import models
import timm

class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""

    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()

        self.num_feature = num_feature
        self.output_size = output_size
        self.linear = nn.Linear(self.num_feature, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.linear(x)


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""

    def __init__(self, num_feature, output_size):
        super(MLP, self).__init__()
        self.hidden = 200
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))

    def forward(self, x):
        return self.model(x)


class MlpModel(nn.Module):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a
    BN layer. Categorical Cross Entropy loss.
    """
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        """
        Returns a new MNISTModelBN.
        """
        super(MlpModel, self).__init__()
        self.in_features = in_features
        self.fc0 = torch.nn.Linear(in_features, hidden_dim)
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 200)
        self.relu1 = torch.nn.ReLU()
        self.out = torch.nn.Linear(200, num_classes)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_layers = [self.bn0]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, self.in_features)
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))

        return self.out(b)


class MnistCNN(nn.Module):
    """from fy"""
    def __init__(self, data_in, data_out):
        super(MnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x


"""from fy"""
class CifarCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(CifarCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(-1, 64 * 4 * 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CifarCNN_MTFL(nn.Module):
    """
    cifar10 model of MTFL
    """

    def __init__(self, data_in, data_out):
        super(CifarCNN_MTFL, self).__init__()

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 10)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]

    def forward(self, x):
        """
        Returns outputs of model given data x.
        Args:
            - x: (torch.tensor) must be on same device as model
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.fc0(self.flat(b)))

        return self.out(c)


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class BasicCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

"""Cluster FL"""
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x


"""FedFomo"""
class BaseConvNet(nn.Module):
    def __init__(self, in_features=1, num_classes=10, ):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNMnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(data_in, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, data_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNFashion_Mnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNCifar(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, data_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# TPDS MTFL model
class CIFAR10Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR10Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, num_classes)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]


        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)

# TPDS MTFL model
class CIFAR100Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR100Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 100)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]
        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)



# from TPDS
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes):
        """
        Returns a new FashionMNISTModel.

        Args:
            - device: (torch.device) to place model on
        """
        super(FashionMNISTModel, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 32, 7, padding=3)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.out = torch.nn.Linear(64 * 7 * 7, num_classes)
        self.bn_layers = [self.bn0, self.bn1]
        self.head = [self.out]
        self.body = [self.conv0,self.bn0,self.conv1,self.bn1]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, 1, 28, 28)
        x = self.bn0(self.pool(self.act(self.conv0(x))))
        x = self.bn1(self.pool(self.act(self.conv1(x))))
        x = x.flatten(1)
        return self.out(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu = torch.nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)
        self.classifier_layer = [self.fc1, self.output]
        self.head = [self.output]
        self.body = [self.conv1,self.conv2,self.fc1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class Reswithoutcon(nn.Module):
    def __init__(self, option='resnet50', pret=False, with_con=True, num_classes=10):
        super(Reswithoutcon, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x


class MobilenetV2(nn.Module):
    def __init__(self, option='MobilenetV2', pret=False, with_con=True,num_classes=10):
        super(MobilenetV2, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        model_ft = models.mobilenet_v2(pretrained=pret,num_classes=num_classes)
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # self.bn_layers = [self.bn0, self.bn1]
        # self.linear_layers = [self.fc0,self.out]
        # self.deep = [self.bn0, self.bn1,self.out]
        # self.shallow = [self.conv0,self.conv1,self.fc0]
        self.head = [self.fc]
        self.body = [self.layer1, self.layer2, self.layer3, self.layer4]

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.fc(out)
        # print(out)
        return out

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        # self.train_step(torch.zeros((2, 3, 64, 64),
        #                             device=self.device,
        #                             dtype=torch.float32),
        #                 torch.zeros((2),
        #                             device=self.device,
        #                             dtype=torch.int32).long())
        pass


def get_mobilenet(num_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param num_classes:
    :return: nn.Module
    """
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

class MobileViT(nn.Module):
    """
    MobileViT 模型封装
    
    GPR 适配：
    - 可选的 GPR 预处理层（信号归一化 + 时空特征增强）
    - 支持冻结 backbone 只训练分类头
    """
    def __init__(self, model_name='mobilevit_s', num_classes=10, pretrained=True, gpr_mode=False):
        super(MobileViT, self).__init__()
        
        self.gpr_mode = gpr_mode
        
        # GPR 专用预处理层
        if gpr_mode:
            self.gpr_preprocess = nn.Sequential(
                # 可学习的信号归一化
                nn.InstanceNorm2d(3, affine=True),
                # 时间域增强（垂直方向）
                nn.Conv2d(3, 16, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                # 空间域增强（水平方向）
                nn.Conv2d(16, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                # 融合回 3 通道
                nn.Conv2d(16, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
            )
        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # FedDWA 需要分离 head 和 body
        # 对于 timm 的 mobilevit，通常 classifier 是 head
        # 我们需要检查具体结构，这里假设是标准的 timm 结构
        
        # 尝试自动识别 head 和 body
        if hasattr(self.model, 'head'):
            self.head = [self.model.head]
            # body 是除了 head 之外的所有部分，这比较难直接获取列表
            # 简单起见，我们把整个 model 当作 features，除了 head
            # 但 FedDWA 需要参数列表。
            # 让我们用 named_children 来区分
            self.body = [m for n, m in self.model.named_children() if n != 'head']
        elif hasattr(self.model, 'classifier'): # MobileNetV3 等
             self.head = [self.model.classifier]
             self.body = [m for n, m in self.model.named_children() if n != 'classifier']
        else:
            # 如果找不到明显的 head，可能需要手动指定，或者把最后的全连接层当作 head
            # 这里做一个通用的 fallback，假设最后一层是 head
            children = list(self.model.children())
            self.head = [children[-1]]
            self.body = children[:-1]

    def forward(self, x):
        if self.gpr_mode:
            x = self.gpr_preprocess(x)
        return self.model(x)

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                for param in bn.parameters():
                    vals.append(copy.deepcopy(param))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                for param in bn.parameters():
                    vals.append(copy.deepcopy(param))
        return vals

    def set_head_val(self, vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                for param in bn.parameters():
                    param.copy_(vals[i])
                    i += 1

    def set_body_val(self, vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                for param in bn.parameters():
                    param.copy_(vals[i])
                    i += 1

try:
    import clip
except ImportError:
    clip = None

class MaskedMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskedMLP, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mask = nn.Parameter(torch.ones(out_features, in_features))
        
    def forward(self, x):
        return F.linear(x, self.mask * self.linear.weight, self.linear.bias)


class GPRAdapter(nn.Module):
    """
    GPR 专用 Adapter 模块
    针对探地雷达信号特点设计的轻量级适配器
    """
    def __init__(self, dim, reduction=4):
        super(GPRAdapter, self).__init__()
        hidden_dim = dim // reduction
        
        # 下投影
        self.down_proj = nn.Linear(dim, hidden_dim)
        # 非线性激活
        self.act = nn.GELU()
        # 上投影
        self.up_proj = nn.Linear(hidden_dim, dim)
        # 可学习的缩放因子（控制 adapter 的影响程度）
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化为近似恒等映射
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        # 残差连接 + adapter
        return x + self.scale * self.up_proj(self.act(self.down_proj(x)))


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, csc=False, class_token_position='end'):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if csc:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(clip_model.device)).type(dtype)

        # Register buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = class_token_position
        self.csc = csc

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            if self.csc:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                 ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = torch.cat(
                [
                    prefix,
                    ctx[:, :half_n_ctx],
                    suffix[:, : -1 - half_n_ctx],
                    ctx[:, half_n_ctx:],
                    suffix[:, -1:],
                ],
                dim=1,
            )
        elif self.class_token_position == "front":
            prompts = torch.cat(
                [
                    prefix,
                    suffix[:, : -1 - self.n_ctx],
                    ctx,
                    suffix[:, -1:],
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts


class FedCLIP(nn.Module):
    """
    FedCLIP: 冻结 CLIP backbone + 可训练 Adapter
    
    GPR 适配：
    - gpr_mode=True 时使用 GPR 专用 Adapter
    - GPR 专用的 prompt 模板
    """
    def __init__(self, model_name='ViT-B/32', device='cuda', num_classes=10, class_names=None, gpr_mode=False, use_coop=False, n_ctx=16, csc=False, class_token_position='end'):
        super(FedCLIP, self).__init__()
        if clip is None:
            raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")
            
        self.device = device
        self.gpr_mode = gpr_mode
        self.use_coop = use_coop
        self.n_ctx = n_ctx
        self.csc = csc
        self.class_token_position = class_token_position
        
        # Load CLIP model
        # jit=False is often recommended for fine-tuning or when using with other torch modules
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval() # Freeze CLIP by default
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize PromptLearner if use_coop is True
        if use_coop and class_names:
             self.prompt_learner = PromptLearner(class_names, self.model, n_ctx=n_ctx, csc=csc, class_token_position=class_token_position)
             self.text_encoder = TextEncoder(self.model)
        else:
             self.prompt_learner = None
             self.text_encoder = None
            
        # Attention Adapter (from FedMedCLIP)
        # CLIP visual output dim: 512 for ViT-B/32, 768 for ViT-L/14
        if model_name == 'ViT-B/32':
            dim = 512
        elif model_name == 'ViT-L/14':
            dim = 768
        else:
            # Infer dim from a dummy run
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(device)
                dim = self.model.encode_image(dummy).shape[1]

        self.dim = dim
        
        if gpr_mode:
            # [优化] GPR 专用 Adapter：更深的结构 + 强正则化防止过拟合
            self.fea_attn = nn.Sequential(
                GPRAdapter(dim, reduction=4),  # 第一层 adapter
                nn.LayerNorm(dim),
                nn.Dropout(0.2),  # [新增] Dropout 防止过拟合
                GPRAdapter(dim, reduction=4),  # 第二层 adapter
                nn.LayerNorm(dim),
                nn.Dropout(0.2),  # [新增] Dropout
            )
            # [优化] GPR 专用分类头：增加深度 + 更强正则化
            self.gpr_classifier = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),  # [调整] 提高 Dropout 比例
                nn.Linear(dim // 2, dim // 4),  # [新增] 增加一层
                nn.ReLU(),
                nn.Dropout(0.2),  # [新增]
                nn.Linear(dim // 4, num_classes),
            )
        else:
            # [优化] 原始 Attention Adapter + 正则化
            self.fea_attn = nn.Sequential(
                MaskedMLP(dim, dim),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2),  # [新增] 防止过拟合
                nn.ReLU(),
                MaskedMLP(dim, dim),
                nn.Dropout(0.2),  # [新增]
                nn.Softmax(dim=1)
            )
        
        self.class_names = class_names
        self.text_features = None
        self.num_classes = num_classes
        
        # Initialize text features if class names are provided
        if self.class_names:
            self.set_class_prompts(self.class_names)
            
    def set_class_prompts(self, class_names):
        self.class_names = class_names
        
        if self.use_coop:
            if self.prompt_learner is None:
                 self.prompt_learner = PromptLearner(class_names, self.model, n_ctx=self.n_ctx, csc=self.csc, class_token_position=self.class_token_position)
                 self.text_encoder = TextEncoder(self.model)
                 self.prompt_learner.to(self.device)
                 self.text_encoder.to(self.device)
            return
        
        # [自定义] GPR 类别描述映射表
        # 为每个类别定义特定的视觉特征描述，增强 CLIP 的理解能力
        custom_gpr_prompts = {
            "Loose": [
                "GPR signal of loose uncompacted soil",
                "low density area in ground penetrating radar",
                "scattered reflections indicating loose material",
                "a region of low compaction in the subsurface"
            ],
            "Crack": [
                "GPR B-scan showing a hyperbolic reflection from a crack",
                "discontinuity in subsurface layers indicating a fracture",
                "vertical crack signature in radargram",
                "a break or fracture in the pavement structure"
            ],
            "Mud Pumping": [
                "GPR signature of mud pumping under pavement",
                "subsurface moisture and fine material accumulation",
                "blurred reflection caused by mud pumping",
                "structural deterioration due to water and soil pumping"
            ],
            "Pipeline": [
                "hyperbolic reflection from a buried pipeline",
                "GPR scan of an underground pipe",
                "inverted U-shape reflection of a utility line",
                "a cylindrical object buried underground"
            ],
            "Redar": [
                "a specific radar anomaly",
                "ground penetrating radar target",
                "distinctive GPR reflection pattern",
                "an identified radar signature"
            ],
            "stell_rib": [
                "strong hyperbolic reflection from a steel rib",
                "GPR image of metal reinforcement bar",
                "regularly spaced high amplitude reflections from steel",
                "metal object embedded in concrete or soil"
            ],
            "Void": [
                "GPR image showing a subsurface void",
                "signal ringing and polarity reversal indicating a cavity",
                "empty space underground in radargram",
                "an air-filled or water-filled hole beneath the surface"
            ],
            "Water Abnormality": [
                "GPR signal attenuation caused by water saturation",
                "high dielectric contrast area indicating water abnormality",
                "subsurface water leakage signature",
                "abnormal moisture content in the ground"
            ]
        }

        # [修正] 始终使用 Text Encoder，即使在 gpr_mode 下
        # 这样可以利用 CLIP 强大的语义锚点能力，避免线性分类头在 Non-IID 下的漂移
        
        # [优化] GPR 专用 prompt 模板集合 (Prompt Ensemble)
        # 使用多种描述方式来增强模型对 GPR 图像特征的理解
        templates = [
            "a ground penetrating radar image showing {}",
            "a GPR B-scan of {}",
            "a radargram containing {}",
            "subsurface detection of {}",
            "a GPR profile with {}",
            "a cross-sectional scan of {}",
            "geophysical data showing {}",
        ]
            
        # 计算每个类别的平均文本特征
        all_text_features = []
        
        with torch.no_grad():
            for c in class_names:
                # 优先使用自定义描述，如果没有则使用模板生成
                if c in custom_gpr_prompts:
                    prompts = custom_gpr_prompts[c]
                else:
                    # 为当前类别生成所有模板的 prompt
                    prompts = [template.format(c) for template in templates]
                
                text_tokens = clip.tokenize(prompts).to(self.device)
                
                # 编码并归一化
                class_embeddings = self.model.encode_text(text_tokens)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
                
                # 平均所有模板的特征 (Prompt Ensembling)
                mean_embedding = class_embeddings.mean(dim=0)
                mean_embedding = mean_embedding / mean_embedding.norm() # 再次归一化
                
                all_text_features.append(mean_embedding)
            
            # 堆叠所有类别的特征 [num_classes, feature_dim]
            # [Fix] Convert to float32 to match image_features in forward pass
            self.text_features = torch.stack(all_text_features).float()
            
    def forward(self, x):
        # Image encoding
        image_features = self.model.encode_image(x).float()
        
        # Adapter
        if self.gpr_mode:
            # [修正] GPR 模式下，我们仍然使用 Text Encoder 作为分类器
            # 但是我们使用更强的 Adapter 来调整图像特征，使其更接近 GPR 的文本描述
            
            # 使用 GPRAdapter 调整图像特征
            adapted_features = self.fea_attn(image_features)
            adapted_features = adapted_features / adapted_features.norm(dim=1, keepdim=True)
            
            # [关键修正] 不再使用 gpr_classifier (线性头)，而是使用 Text Features
            # 这样可以利用 CLIP 的 Zero-shot 能力作为强先验，防止过拟合
            
            if self.use_coop and self.prompt_learner is not None:
                prompts = self.prompt_learner()
                tokenized_prompts = self.prompt_learner.tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            else:
                text_features = self.text_features

            if text_features is None:
                 if self.training:
                     raise ValueError("Class prompts not set. Call set_class_prompts() first.")
                 return torch.zeros(x.size(0), self.num_classes).to(self.device)
            
            # 计算图像特征与文本特征的相似度
            logit_scale = self.model.logit_scale.exp().float()
            logits = logit_scale * adapted_features @ text_features.t()
            
        else:
            # 原始模式：使用 attention adapter + text similarity
            attn_weights = self.fea_attn(image_features)
            image_features = torch.mul(attn_weights, image_features)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Classification (Similarity with text features)
            if self.use_coop and self.prompt_learner is not None:
                prompts = self.prompt_learner()
                tokenized_prompts = self.prompt_learner.tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            else:
                text_features = self.text_features

            if text_features is None:
                 # If no text features, we can't classify. 
                 # For now, raise error or return dummy if just initializing
                 if self.training:
                     raise ValueError("Class prompts not set. Call set_class_prompts() first.")
                 return torch.zeros(x.size(0), self.num_classes).to(self.device)
                 
            # [Fix] Convert logit_scale to float32
            logit_scale = self.model.logit_scale.exp().float()
            logits = logit_scale * image_features @ text_features.t()
        
        return logits
        
    # FedDWA interfaces
    def get_head_val(self):
        # For FedCLIP, the "head" is the adapter we are training
        vals = []
        with torch.no_grad():
            for param in self.fea_attn.parameters():
                vals.append(copy.deepcopy(param))
            
            if self.use_coop and self.prompt_learner is not None:
                for param in self.prompt_learner.parameters():
                    vals.append(copy.deepcopy(param))
                    
        return vals
        
    def set_head_val(self, vals):
        i = 0
        with torch.no_grad():
            for param in self.fea_attn.parameters():
                param.copy_(vals[i])
                i += 1
            
            if self.use_coop and self.prompt_learner is not None:
                for param in self.prompt_learner.parameters():
                    param.copy_(vals[i])
                    i += 1
                
    def get_body_val(self):
        # Body is frozen
        return []

    def set_body_val(self, vals):
        pass


# ============================================================================
# GPR-FedSense: 专为探地雷达数据设计的联邦学习架构
# ============================================================================

class GPRSignalNorm(nn.Module):
    """
    GPR 信号归一化层
    可学习的归一化参数，适配不同设备/环境的信号特性
    """
    def __init__(self, num_features):
        super(GPRSignalNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 可学习的信号增益校正
        self.gain = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # 实例归一化 (适配单样本的设备差异)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x * self.gamma + self.beta
        x = x * self.gain
        return x


class GPRFeatureExtractor(nn.Module):
    """
    GPR 专用特征提取器
    结合 1D（时间域）和 2D（空间域）卷积，捕获 GPR 信号的时频特征
    
    设计理念：
    - 浅层：1D 卷积提取时间域反射特征
    - 中层：2D 卷积提取空间结构特征
    - 深层：混合注意力增强关键区域
    """
    def __init__(self, in_channels=3, base_dim=64):
        super(GPRFeatureExtractor, self).__init__()
        
        # 可学习的信号归一化（适配不同设备）
        self.signal_norm = GPRSignalNorm(in_channels)
        
        # Stage 1: 浅层特征（捕获边缘和纹理）
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: 时间域特征（垂直方向卷积，捕获深度反射）
        self.time_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # Stage 3: 空间域特征（水平方向卷积，捕获横向延续性）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
        )
        
        # 输出维度
        self.out_channels = base_dim * 2
        
    def forward(self, x):
        # 信号归一化
        x = self.signal_norm(x)
        
        # Stage 1
        x = self.stage1(x)
        
        # 并行的时间/空间特征提取
        time_feat = self.time_conv(x)
        spatial_feat = self.spatial_conv(x)
        
        # 特征融合
        x = torch.cat([time_feat, spatial_feat], dim=1)
        x = self.fusion(x)
        
        return x


class GPRFedModel(nn.Module):
    """
    GPR-FedSense: 探地雷达联邦学习专用模型
    
    架构特点：
    1. 本地私有层：GPR 信号归一化 + 特征提取（适配不同设备/环境）
    2. 全局共享层：深层特征提取（跨客户端知识共享）
    3. 个性化分类头：ALA 自适应聚合（处理 Non-IID）
    
    Args:
        num_classes: 分类类别数
        base_dim: 基础通道数
        backbone: 共享层 backbone 类型 ('cnn', 'resnet18', 'mobilevit')
        pretrained: 是否使用预训练权重
    """
    def __init__(self, num_classes=8, base_dim=64, backbone='cnn', pretrained=True, image_size=224):
        super(GPRFedModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_type = backbone
        
        # ============ 模块 1: GPR 本地特征提取器 (私有，不聚合) ============
        self.local_extractor = GPRFeatureExtractor(in_channels=3, base_dim=base_dim)
        local_out_dim = self.local_extractor.out_channels  # 128
        
        # ============ 模块 2: 共享 Backbone (全局聚合) ============
        if backbone == 'cnn':
            self.shared_backbone = nn.Sequential(
                nn.Conv2d(local_out_dim, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feature_dim = 512
            
        elif backbone == 'resnet18':
            # 使用 ResNet18，但替换第一层以接收 local_extractor 的输出
            resnet = models.resnet18(pretrained=pretrained)
            resnet.conv1 = nn.Conv2d(local_out_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 移除原始的 fc 层
            self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 512
            
        elif backbone == 'mobilevit':
            # 使用 MobileViT，但需要适配输入通道
            self.adapter_conv = nn.Conv2d(local_out_dim, 3, kernel_size=1)  # 转换回 3 通道
            self.shared_backbone = timm.create_model('mobilevitv2_050', pretrained=pretrained, num_classes=0)
            feature_dim = self.shared_backbone.num_features
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        self.feature_dim = feature_dim
        
        # ============ 模块 3: 个性化分类头 (本地微调 + ALA) ============
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        
        # 用于 FedDecorr 的特征输出钩子
        self.features = None
        
    def forward(self, x, return_features=False):
        # 本地特征提取
        x = self.local_extractor(x)
        
        # 共享 backbone
        if self.backbone_type == 'mobilevit':
            x = self.adapter_conv(x)
        x = self.shared_backbone(x)
        
        # 展平
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 保存特征用于 FedDecorr
        self.features = x
        
        # 分类
        out = self.classifier(x)
        
        if return_features:
            return out, x
        return out
    
    def get_features(self):
        """获取最后一层特征，用于 FedDecorr"""
        return self.features
    
    # ============ FedDWA 接口 ============
    def get_head_val(self):
        """获取分类头参数（用于个性化聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.classifier.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_head_val(self, vals):
        """设置分类头参数"""
        i = 0
        with torch.no_grad():
            for param in self.classifier.parameters():
                param.copy_(vals[i])
                i += 1
                
    def get_body_val(self):
        """获取共享层参数（用于全局聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.shared_backbone.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_body_val(self, vals):
        """设置共享层参数"""
        i = 0
        with torch.no_grad():
            for param in self.shared_backbone.parameters():
                param.copy_(vals[i])
                i += 1
                
    def get_local_val(self):
        """获取本地私有层参数（不参与聚合）"""
        vals = []
        with torch.no_grad():
            for param in self.local_extractor.parameters():
                vals.append(copy.deepcopy(param))
        return vals
    
    def set_local_val(self, vals):
        """设置本地私有层参数"""
        i = 0
        with torch.no_grad():
            for param in self.local_extractor.parameters():
                param.copy_(vals[i])
                i += 1
