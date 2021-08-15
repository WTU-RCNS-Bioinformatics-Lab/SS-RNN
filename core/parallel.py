import os
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '123455'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print("Rank {}/{} process initialized.".format(rank+1, world_size))


def cleanup():
    dist.destroy_process_group()