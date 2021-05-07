import argparse
import os
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from easydict import EasyDict
from interface.temporal_helper import TemporalHelper

# import logging

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='X-Temporal')
parser.add_argument('--config', type=str, help='the path of config file')
parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
                    default=0, type=int)
parser.add_argument("--num_shards", help="Number of shards using by the job",
                    default=2, type=int)
parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
                    default="env://", type=str)
parser.add_argument('--dist_backend', default='nccl', type=str)

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--train', dest='train', action='store_true')

# ----------------------------------------------------------------------------------------------------------------
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config['config'])

# print(os.environ)
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method="env://")
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
else:
    args.world_size = 1
    args.rank = 0

args.total_batch_size = config.dataset.batch_size * args.world_size

# init cuda env
cudnn.benchmark = True
# torch.cuda.set_device(args.local_rank)


if args.train:
    temporal_helper = TemporalHelper(config)
    temporal_helper.train()
else:
    temporal_helper = TemporalHelper(config, inference_only=True)
    temporal_helper.evaluate()
