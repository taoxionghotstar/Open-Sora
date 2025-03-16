import math
import random
from collections import OrderedDict

import torch
import torch.distributed as dist
from colossalai.booster.plugin import LowLevelZeroPlugin

from opensora.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group
from opensora.acceleration.plugin import ZeroSeqParallelPlugin

from .misc import get_logger


def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20):
    ## 使用ZeRO阶段2的优化，支持内存优化和混合精度训练，不支持序列并行（sp_size必须为1）
    if plugin == "zero2":
        assert sp_size == 1, "Zero2 plugin does not support sequence parallelism"
        ## LowLevelZeroPlugin用于实现ZeRO（Zero Redundancy Optimizer）的第1阶段和第2阶段优化
        ## ZeRO阶段1，切分优化器状态（如梯度、动量等），并将它们分发到各个数据并行进程或GPU上；提供比PyTorch的DistributedDataParallel（DDP）更高的内存效率和更快的训练速度
        ## ZeRO阶段2，在阶段1的基础上，进一步切分优化器状态和梯度；不支持局部梯度累积，因此不建议与流水线并行（Pipeline Parallelism）一起使用￼。
        plugin = LowLevelZeroPlugin(
            stage=2,  ## 使用ZeRO的第二阶段优化
            precision=dtype,
            initial_scale=2**16,  ## 初始缩放因子，用于混合精度训练
            max_norm=grad_clip,  ## 梯度裁剪的最大范数
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,  ## 梯度归约的桶大小（单位为百万）。较大的桶大小可以减少通信次数，但会增加内存占用【
        )
        ## 设置当前的数据并行组，以便在后续的训练过程中，ColossalAI可以正确地管理数据并行的通信和同步操作
        ## dist.group.WORLD是一个全局进程组，表示所有已初始化的进程，包含所有参与训练的进程
        set_data_parallel_group(dist.group.WORLD)  ## 所有进程都将参与数据并行操作
    elif plugin == "zero2-seq":
        assert sp_size > 1, "Zero2-seq plugin requires sequence parallelism"
        ## ZeroSeqParallelPlugin用于在大规模分布式训练中实现高效的内存优化和并行计算
        plugin = ZeroSeqParallelPlugin(
            sp_size=sp_size,  ## 序列并行的大小，表示将输入序列分割成多少个子序列
            stage=2,  ## 使用阶段2，表示切分优化器状态和梯度
            precision=dtype,
            initial_scale=2**16,  ## 初始缩放因子，用于混合精度训练
            max_norm=grad_clip,  ## 梯度裁剪的最大范数
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,  ## 梯度归约的桶大小（单位为百万）。较大的桶大小可以减少通信次数，但会增加内存占用
        )
        ## 设置序列并行（Sequence Parallelism）进程组，将插件中定义的序列并行组（sp_group）设置为当前的序列并行进程组
        set_sequence_parallel_group(plugin.sp_group)
        ## 设置当前的数据并行组，以便在后续的训练过程中，ColossalAI可以正确地管理数据并行的通信和同步操作
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        get_logger().info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks
