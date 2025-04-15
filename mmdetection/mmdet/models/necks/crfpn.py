from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from functools import reduce

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, times=2):
        super(Downsample, self).__init__()
        out_channels = in_channels if out_channels == None else out_channels
        self.downsample = ConvModule(in_channels, out_channels, times, times, 0, )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale_factor=2.0, mode='nearest'):
        super(Upsample, self).__init__()
        out_channels = in_channels if out_channels == None else out_channels
        self.upsample = nn.Sequential(
            ConvModule(in_channels, out_channels, 1, 1, 0),
            nn.Upsample(scale_factor=scale_factor, mode=mode))

    def forward(self, x):
        x = self.upsample(x)
        return x


class AdaptiveChannelfusion(nn.Module):
    def __init__(self, in_channels, num_input=2, ratio=16, L=32, norm_cfg=None):
        """
        """
        super(AdaptiveChannelfusion, self).__init__()
        self.in_channels = in_channels
        self.num_input = num_input
        self.inter_dim = max(in_channels // ratio, L)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            ConvModule(self.in_channels, self.inter_dim, 1, 1, 0, bias=False),
            ConvModule(self.inter_dim, self.in_channels * num_input, 1, 1, 0, bias=False, act_cfg=None))
        self.conv = ConvModule(self.in_channels, self.in_channels, 3, 1, 1, norm_cfg=norm_cfg)

    def forward(self, input):
        assert len(input) == self.num_input
        batch_size = input[0].size(0)
        fused = reduce(lambda x, y: x + y, input)
        w = self.fc(self.global_pool(fused))
        w = F.softmax(w.reshape(batch_size, self.num_input, self.in_channels, -1), dim=1)
        w = list(w.chunk(self.num_input, dim=1))
        outputs = []
        for i in range(self.num_input):
            outputs.append(input[i] * w[i].reshape(batch_size, self.in_channels, 1, 1))
        output = self.conv(reduce(lambda x, y: x + y, outputs))

        return output


class AdaptiveSpatialfusion(nn.Module):
    def __init__(self, in_channels, num_inputs=2, norm_cfg=None):
        super(AdaptiveSpatialfusion, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        compress_c = 8
        self.compress_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.compress_layers.append(ConvModule(self.in_channels, compress_c, 1, 1, 0))

        self.weight_levels = nn.Conv2d(compress_c * self.num_inputs, self.num_inputs, kernel_size=1, stride=1,
                                       padding=0)
        self.conv = ConvModule(self.in_channels, self.in_channels, 3, 1, 1, norm_cfg=norm_cfg)

    def forward(self, input):
        compress_layers_output = [lateral_conv(input[i])
                                  for i, lateral_conv in enumerate(self.compress_layers)]
        levels_weight_v = torch.cat(compress_layers_output, dim=1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        outputs = []
        for i in range(self.num_inputs):
            outputs.append(input[i] * levels_weight[:, i:i + 1, :, :])
        out = self.conv(reduce(lambda j, k: j + k, outputs))
        return out


class WeightAdd(nn.Module):
    def __init__(self, c1, c2=None, num_input=2, norm_cfg=None):
        super(WeightAdd, self).__init__()
        c2 = c1 if c2 is None else c2
        self.num_input = num_input
        self.w = nn.Parameter(torch.ones(self.num_input, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.relu = nn.ReLU(inplace=False)
        self.hswish = nn.Hardswish()
        self.conv = ConvModule(c1, c2, kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, inputs):
        w = self.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        outputs = []
        for i in range(self.num_input):
            outputs.append(weight[i] * inputs[i])
        output = reduce(lambda x, y: x + y, outputs)
        return self.conv(self.hswish(output))


class NeckBasicBody(nn.Module):
    def __init__(self, inter_dim, norm_cfg=None, ):
        super(NeckBasicBody, self).__init__()
        if isinstance(inter_dim, list):
            ch_1, ch_2, ch_3, ch_4, ch_5 = inter_dim
        elif isinstance(inter_dim, int):
            ch_1, ch_2, ch_3, ch_4, ch_5 = [inter_dim for _ in range(5)]
        # stage 1
        # for e
        self.e_dw = Downsample(ch_4, ch_5, times=2)
        # for c
        self.c_up = Upsample(ch_4, ch_3, scale_factor=2)
        self.c_dw = Downsample(ch_2, ch_3, times=2)
        # for a
        self.a_up = Upsample(ch_2, ch_1, scale_factor=2)

        # stage 2
        # for d
        self.d_up = Upsample(ch_5, ch_4, scale_factor=2)
        self.d_dw = Downsample(ch_3, ch_4, times=2)
        # for b
        self.b_up = Upsample(ch_3, ch_2, scale_factor=2)
        self.b_dw = Downsample(ch_1, ch_2, times=2)

        # weightAdd p layers
        self.e_WAdd = WeightAdd(ch_5, num_input=2, norm_cfg=norm_cfg)
        self.d_WAdd = WeightAdd(ch_4, num_input=3, norm_cfg=norm_cfg)
        self.c_WAdd = WeightAdd(ch_3, num_input=3, norm_cfg=norm_cfg)
        self.b_WAdd = WeightAdd(ch_2, num_input=3, norm_cfg=norm_cfg)
        self.a_WAdd = WeightAdd(ch_1, num_input=2, norm_cfg=norm_cfg)

    def forward(self, inputs):
        a, b, c, d, e = inputs
        # stage 1
        # a, c, e
        e = self.e_WAdd((e, self.e_dw(d)))
        # import pdb
        # pdb.set_trace()
        c = self.c_WAdd((self.c_up(d), c, self.c_dw(b)))
        a = self.a_WAdd((self.a_up(b), a))

        # stage 2
        # b, d
        d = self.d_WAdd((self.d_up(e), d, self.d_dw(c)))
        b = self.b_WAdd((self.b_up(c), b, self.b_dw(a)))

        return tuple([a, b, c, d, e])


class NeckHead(nn.Module):
    def __init__(self, inter_dim, norm_cfg=None):
        super(NeckHead, self).__init__()
        if isinstance(inter_dim, list):
            ch_1, ch_2, ch_3, ch_4, ch_5 = inter_dim
        elif isinstance(inter_dim, int):
            ch_1, ch_2, ch_3, ch_4, ch_5 = [inter_dim for _ in range(5)]
        # acf on c, d, e
        self.acf = AdaptiveChannelfusion(ch_4, num_input=3, norm_cfg=norm_cfg)
        self.acf_up = Upsample(ch_5, ch_4, scale_factor=2)
        self.acf_dw = Downsample(ch_3, ch_4, times=2)

        # asf on a, b, c
        self.asf = AdaptiveSpatialfusion(ch_2, num_inputs=3, norm_cfg=norm_cfg)
        self.asf_up = Upsample(ch_3, ch_2, scale_factor=2)
        self.asf_dw = Downsample(ch_1, ch_2, times=2)

    def forward(self, inputs):
        a, b, c, d, e = inputs
        # acf
        acf_out = self.acf((self.acf_up(e), d, self.acf_dw(c)))
        # asf
        asf_out = self.asf((self.asf_up(c), b, self.asf_dw(a)))

        d = d + acf_out
        b = b + asf_out

        return b, d


@MODELS.register_module()
class CRFPN(nn.Module):
    def __init__(self, in_channels: List[int] = [256, 512, 1024, 2048],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 stack_times: int = 2,
                 start_level: int = 0,
                 end_level: int = -1,
                 norm_cfg: OptConfigType = None,
                 ):
        super(CRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.stack_times = stack_times
        self.norm_cfg = norm_cfg

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvModule(in_channels[i], out_channels, 1, norm_cfg=norm_cfg, act_cfg=None))

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            self.extra_downsamples.append(nn.Sequential(
                ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None),
                nn.MaxPool2d(2, 2)))

        # NeckHead
        self.neckhead = NeckHead(self.out_channels, norm_cfg=self.norm_cfg)

        # BasicBody
        self.neck_basicbody = nn.Sequential(
            *[NeckBasicBody(inter_dim=self.out_channels, norm_cfg=self.norm_cfg) \
              for _ in range(self.stack_times)]
        )

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build a-c
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build d-e on top of c
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        a, b, c, d, e = feats

        # NeckHead
        b, d = self.neckhead((a, b, c, d, e))

        # Basicbody
        outputs = self.neck_basicbody((a, b, c, d, e))

        return tuple(outputs)
