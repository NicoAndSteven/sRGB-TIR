import torch
import torch.nn as nn
import torch.nn.functional as F


class HED(nn.Module):
    def __init__(self, pretrained=False, model_path=None):
        super(HED, self).__init__()

        # 定义与官方一致的网络结构
        self.netVggOne = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netScoreOne = nn.Conv2d(64, 1, kernel_size=1)
        self.netScoreTwo = nn.Conv2d(128, 1, kernel_size=1)
        self.netScoreThr = nn.Conv2d(256, 1, kernel_size=1)
        self.netScoreFou = nn.Conv2d(512, 1, kernel_size=1)
        self.netScoreFiv = nn.Conv2d(512, 1, kernel_size=1)

        self.netCombine = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 加载本地预训练权重
        if pretrained:
            self.load_pretrained(model_path)

    def forward(self, x):
        # 官方前向传播逻辑
        x = x * 255.0
        x = x - torch.tensor([104.00698793, 116.66876762, 122.67891434],
                             dtype=x.dtype, device=x.device).view(1, 3, 1, 1)

        feat1 = self.netVggOne(x)
        feat2 = self.netVggTwo(feat1)
        feat3 = self.netVggThr(feat2)
        feat4 = self.netVggFou(feat3)
        feat5 = self.netVggFiv(feat4)

        # 侧边输出
        score1 = self.netScoreOne(feat1)
        score2 = self.netScoreTwo(feat2)
        score3 = self.netScoreThr(feat3)
        score4 = self.netScoreFou(feat4)
        score5 = self.netScoreFiv(feat5)

        # 上采样到原图尺寸
        score2 = F.interpolate(score2, size=x.shape[2:], mode='bilinear', align_corners=False)
        score3 = F.interpolate(score3, size=x.shape[2:], mode='bilinear', align_corners=False)
        score4 = F.interpolate(score4, size=x.shape[2:], mode='bilinear', align_corners=False)
        score5 = F.interpolate(score5, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 融合输出
        fused = self.netCombine(torch.cat([score1, score2, score3, score4, score5], dim=1))
        return fused

    def load_pretrained(self, model_path):
        # 加载本地权重（需与官方结构严格匹配）
        state_dict = torch.load(model_path, map_location='cpu')

        # 适配层名称（如果官方代码使用"netVggOne"，而你的代码使用"conv1"）
        state_dict = {k.replace('module', 'net'): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded HED pretrained weights from {model_path}")