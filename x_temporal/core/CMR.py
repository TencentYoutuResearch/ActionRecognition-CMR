import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SME(nn.Module):
    def __init__(self, channels, n_segment):
        super(SME, self).__init__()
        self.channels = channels
        self.n_segment = n_segment
        self.W = nn.Sequential(
            nn.Conv2d(self.channels, self.channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W[1].weight.data, 0.0)
        nn.init.constant_(self.W[1].bias.data, 0.0)

    def forward(self, x):
        nt, c, h, w = x.size()
        # x_p = self.pool(x)
        x_p = x
        n_batch = nt // self.n_segment
        x_p = x_p.view(n_batch, self.n_segment, c, h, w)
        # x = x.view(n_batch, self.n_segment, c, h, w)
        x_left = x_p[:, :-1]
        x_right = x_p[:, 1:]
        x_left = x_left.reshape(n_batch * (self.n_segment - 1), c, h * w)
        x_right = x_right.reshape(n_batch * (self.n_segment - 1), c, h * w)
        simi = F.cosine_similarity(x_left, x_right, dim=1)
        simi = simi.reshape(n_batch, self.n_segment - 1, h, w)
        simi = torch.cat([simi, simi[:, -1].unsqueeze(1)], dim=1)
        x_p = x_p * (1 - simi.unsqueeze(2))
        x_p = x_p.view(nt, c, h, w)
        x = self.W(x_p) + x
        # x = self.W(x_p+x)
        return x  # nt,c,h,w


class CME(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment, kernel_size, stride=1, padding=0, bias=True, n_div=8):
        super(CME, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.n_div = n_div

        self.conv1_reduce_q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // n_div, kernel_size=1, stride=1, padding=padding, bias=bias),
            nn.BatchNorm2d(in_channels // n_div),
            nn.ReLU(inplace=True))
        self.conv1_reduce_k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // n_div, kernel_size=1, stride=1, padding=padding, bias=bias),
            nn.BatchNorm2d(in_channels // n_div),
            nn.ReLU(inplace=True))

        self.conv_inflate = nn.Sequential(
            nn.Conv2d(in_channels // n_div, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def Trans_att(self, q_matrix, k_matrix, v_matrix):
        '''
        Args:
            q_matrix:
            k_v_matrix:

        Returns:

        '''
        nt, _, _, _ = q_matrix.size()
        n_batch = nt // self.n_segment
        q_matrix = q_matrix.view(n_batch, self.n_segment, -1)
        k_matrix = k_matrix.view(n_batch, self.n_segment, -1)
        v_matrix = v_matrix.view(n_batch, self.n_segment, -1)
        ATT = torch.matmul(q_matrix, k_matrix.transpose(1, 2))
        ATT = -ATT
        ATT = F.softmax(ATT, 1)
        Q_update = torch.matmul(ATT, v_matrix) + v_matrix
        return Q_update

    def forward(self, x):
        nt, c, h, w = x.size()
        out_q = F.adaptive_avg_pool2d(self.conv1_reduce_q(x), (1, 1))
        out_k = F.adaptive_avg_pool2d(self.conv1_reduce_k(x), (1, 1))
        out_v = out_k
        Q_encoding = self.Trans_att(out_q, out_q, out_v)
        Q_encoding = Q_encoding.view(nt, c // self.n_div, 1, 1)
        out = Q_encoding
        out = self.conv_inflate(out)
        out = torch.sigmoid(out)
        return out * x


class TIM(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=True, n_div=8, p_init_type='tsm'):
        super(TIM, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.fold_div = n_div
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        conv_params = torch.zeros((in_channels, 1) + kernel_size)
        fold = in_channels // n_div

        # TSM initialization
        if p_init_type == 'r_tsm':
            for i in range(in_channels):
                import random
                j = random.randint(0, kernel_size[0] - 1)
                conv_params[i, :, j] = 1
            self.weight = nn.Parameter(conv_params)
        elif p_init_type == 'tsm':
            conv_params[:fold, :, kernel_size[0] // 2 + 1] = 1
            conv_params[fold:2 * fold, :, kernel_size[0] // 2 - 1] = 1
            conv_params[2 * fold:, :, kernel_size[0] // 2] = 1
            self.weight = nn.Parameter(conv_params)
        elif p_init_type == 'TSN':
            conv_params[:, :, kernel_size[0] // 2] = 1
            self.weight = nn.Parameter(conv_params)
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(4))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding, 1, self.in_channels)


class TemporalGlobal(nn.Module):
    def __init__(self, net, n_segment=3, has_att=False, n_div=8, shift_kernel=3, shift_grad=0.0, has_spatial_att=False):
        super(TemporalGlobal, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.has_att = has_att
        self.fold_div = n_div
        self.has_spatial_att = has_spatial_att
        if has_att:
            self.trans_att = CME(net.conv1.in_channels, net.conv1.in_channels, n_segment, (1, 1), padding=0,
                                 n_div=n_div, bias=False)
        self.shift_conv = TIM(net.conv1.in_channels, (shift_kernel, 1), padding=(shift_kernel // 2, 0),
                              n_div=n_div, bias=False)
        if has_spatial_att:
            self.spatial_att = SME(net.conv3.out_channels, self.n_segment)

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        identity = x

        if self.has_att:
            out = self.trans_att(x)
        else:
            out = x
        reshape_x = out.view(n_batch, -1, c, h * w).permute(0, 2, 1, 3).contiguous()
        shift_x = self.shift_conv(reshape_x)
        shift_x = shift_x.permute(0, 2, 1, 3).contiguous().view(nt, c, h, w)
        out = self.net.conv1(shift_x)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)

        if self.has_spatial_att:
            out = self.spatial_att(out)

        return out


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, has_att=False, n_div=8, place='blockres', shift_type='iCover', shift_kernel=3,
                        shift_grad=0.0, temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    if 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2

        def make_block_temporal(stage, this_segment, has_att, spatial_att=True):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i == 0:
                    blocks[i] = TemporalGlobal(b, n_segment=this_segment, has_att=has_att, n_div=n_div,
                                               shift_grad=shift_grad, has_spatial_att=(True and spatial_att))
                elif i % n_round == 0:
                    blocks[i] = TemporalGlobal(b, n_segment=this_segment, has_att=has_att, n_div=n_div,
                                               shift_grad=shift_grad, has_spatial_att=(False and spatial_att))
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], has_att)
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], has_att)
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], has_att)
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], has_att)
    else:
        raise NotImplementedError


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError
