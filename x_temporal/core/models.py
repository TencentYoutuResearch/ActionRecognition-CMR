import sys

sys.path.append("..")
from torch import nn

from core.basic_ops import ConsensusModule
from core.transforms import *
from torch.nn.init import normal_, constant_
from models.resnet import *

import logging

logger = logging.getLogger('global')


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain=True,
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False, non_local=False, has_att=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.img_feature_dim = img_feature_dim
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place

        self.base_model_name = base_model
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        self.has_att = has_att

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" or modality == "audio" else 5
        else:
            self.new_length = new_length
        if print_spec:
            logger.info(("""
    Initializing with base model: {}.
    Model Configurations:
          input_modality:     {}
          num_segments:       {}
          new_length:         {}
          consensus_module:   {}
          dropout_ratio:      {}
          img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Linear(
                    feature_dim,
                    num_class))
            self.new_fc = None
        else:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(
                    p=self.dropout))
            if self.consensus_type in ['TRN', 'TRNmultiscale']:
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(
                    self.base_model,
                    self.base_model.last_layer_name).weight,
                0,
                std)
            constant_(
                getattr(
                    self.base_model,
                    self.base_model.last_layer_name).bias,
                0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        logger.info('=> base model: {}'.format(base_model))

        if base_model.startswith('resnet'):
            self.base_model = globals()[base_model](pretrained=self.pretrain)
            from core.CMR import make_temporal_shift
            logger.info('prepare CMR modeling!!!')
            make_temporal_shift(self.base_model, self.num_segments, has_att=self.has_att,
                                n_div=self.shift_div, temporal_pool=self.temporal_pool, shift_grad=1.0)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            logger.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input, no_reshape=False, extract_feat=False):
        if not no_reshape:
            sample_len = 3
            base_out, _ = self.base_model(input.view(
                (-1, sample_len) + input.size()[-2:]))
        else:
            base_out, _ = self.base_model(input)
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view(
                    (-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view(
                    (-1, self.num_segments) + base_out.size()[1:])
                base_out = base_out
            output = self.consensus(base_out)
            return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                logger.info('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        else:
            raise NotImplementedError
