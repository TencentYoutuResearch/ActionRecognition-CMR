import sys

sys.path.append("..")
from core.models import TSN
from core.transforms import *
from core.CMR import *
import torchvision


def get_model(config):
    num_class = config.dataset.num_class

    if config.net.model_type == '2D':
        model = TSN(num_class, config.dataset.num_segments, config.dataset.modality,
                    base_model=config.net.arch,
                    consensus_type=config.net.consensus_type,
                    dropout=config.net.dropout,
                    img_feature_dim=config.net.img_feature_dim,
                    partial_bn=not config.trainer.no_partial_bn,
                    is_shift=config.net.get('shift', False), shift_div=config.net.get('shift_div', 8),
                    non_local=config.net.get('non_local', False),
                    pretrain=config.net.get('pretrain', True),
                    )
    else:
        raise ValueError("Not Found model type: %s" % config.net.model_type)

    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if config.gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True)

    return model


def get_augmentation(config):
    if config.dataset.modality == 'RGB':
        if config.dataset.flip:
            return torchvision.transforms.Compose([
                GroupScale(size=(256, 340)),
                GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose(
                [
                    GroupScale(size=(256, 340)),
                    GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75, .66])])
    elif config.dataset.modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])
    elif config.dataset.modality == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(config.dataset.crop_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=False)])
    elif config.dataset.modality == 'audio':
        return torchvision.transforms.Compose([GroupScale((config.dataset.crop_size, config.dataset.crop_size))])
