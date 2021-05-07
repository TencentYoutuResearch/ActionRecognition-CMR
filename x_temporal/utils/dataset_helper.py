import sys

sys.path.append("..")

from core.transforms import *
from core.dataset import VideoDataSet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


def get_val_crop_transform(config, spatial_crops):
    crop_size = config.crop_size
    scale_size = config.scale_size
    if spatial_crops == 1:
        crop_aug = GroupCenterCrop(crop_size)
    elif spatial_crops == 3:
        crop_aug = GroupFullResSample(crop_size, scale_size, flip=False)
    elif spatial_crops == 5:
        crop_aug = GroupOverSample(crop_size, scale_size, flip=False)
    else:
        crop_aug = MultiGroupRandomCrop(crop_size, spatial_crops)
    return crop_aug


def get_dataset(config, data_type="train", test_mode=False, transform=None, data_length=1, temporal_samples=1):
    dataset = VideoDataSet(config.root_dir, config[data_type].meta_file,
                           num_segments=config.num_segments,
                           new_length=data_length,
                           modality=config.modality,
                           image_tmpl=config.img_prefix,
                           test_mode=False,
                           random_shift=not test_mode,
                           transform=transform,
                           dense_sample=config.dense_sample,
                           dense_sample_rate=config.get('dense_sample_rate', 1),
                           video_source=config.video_source,
                           temporal_samples=temporal_samples,
                           multi_class=config.get('multi_class', False),
                           test_online=(data_type == 'test')
                           )
    return dataset


def shuffle_dataset(loader, cur_epoch):
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
