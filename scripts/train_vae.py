import os
import random
from datetime import timedelta
from pprint import pformat

import torch
## torch.distributed是PyTorch中用于支持分布式训练的包，提供了多进程并行计算的通信原语，可以跨多个计算节点（包括单机多GPU或多机多GPU）进行训练
import torch.distributed as dist
## wandb（Weights & Biases）是一个用于机器学习和深度学习实验跟踪、可视化和管理的工具
import wandb
## ColossalAI是一个用于大规模分布式训练和高效深度学习的开源框架，旨在帮助用户更轻松地实现高性能的分布式训练
## Booster是ColossalAI中的高级封装工具，用于简化分布式训练的配置和执行，将模型、优化器、数据加载器等组件封装在一起，提供了一种“开箱即用”的方式来启动分布式训练
from colossalai.booster import Booster
## DistCoordinator是ColossalAI中的一个组件，用于管理和协调分布式训练环境中的进程和资源
## 它提供了对分布式集群的高级控制，包括任务调度、资源分配、进程管理等功能，使得用户可以更方便地管理和监控大规模分布式训练任务
from colossalai.cluster import DistCoordinator
## HybridAdam是ColossalAI提供的一种优化器，它是对PyTorch中经典的Adam优化器的扩展和改进
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
## einops（Einstein Operations）是一个用于张量操作的Python库，提供了简洁而强大的语法来处理张量的重塑、排列和变换
## rearrange函数允许你通过简单的字符串语法来描述张量的维度变换
from einops import rearrange
## TQDM（Text Progress Bar for Python）是一个流行的Python库，用于在终端或Jupyter Notebook中显示进度条
## 它可以帮助开发者在长时间运行的任务中直观地跟踪进度，提升用户体验
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.models.vae.losses import AdversarialLoss, DiscriminatorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt_utils import load, save
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.misc import (
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    to_torch_dtype,
)
from opensora.utils.train_utils import create_colossalai_plugin


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    ## config file: configs/vae/train/stage1.py, stage2.py, stage3.py
    cfg = parse_configs(training=True)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")  ## "bf16"
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    ## dist是PyTorch分布式训练模块，init_process_group用于初始化进程组（process group）
    ## 它会设置一个进程组，使得多个进程能够通过指定的通信后端（如NCCL、Gloo或MPI）进行通信
    ## NCCL是一个专为NVIDIA GPU设计的高性能通信库，常用于GPU之间的通信
    ## 设置进程组通信的超时时间，用于防止某些进程提前退出，导致分布式训练中断；这里设置为24小时，以避免意外超时
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    ## dist.get_rank()返回当前进程的全局排名（rank），每个进程在分布式环境中都有一个唯一的排名
    ## torch.cuda.device_count()返回系统中可用的GPU数量
    ## dist.get_rank() % torch.cuda.device_count()确保每个进程分配到不同的GPU，避免多个进程同时使用同一个GPU导致冲突
    ## torch.cuda.set_device()设置当前进程使用的GPU设备
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))  ## 用于设置随机种子，以确保训练过程的可重复性，“seed"为42
    ## DistCoordinator用于管理分布式训练中的节点和进程，它可以帮助用户在分布式环境中协调资源分配、任务调度和通信
    ## DistCoordinator() 被实例化为一个对象 coordinator，这通常是分布式训练流程的入口点
    ## 通过coordinator对象可以访问和管理集群中的节点、进程以及执行分布式任务
    coordinator = DistCoordinator()
    ## get_current_device()用于获取当前环境中默认的计算设备
    ## 通常用于分布式训练环境，确保代码能够正确地在 GPU 或其他设备上运行
    device = get_current_device()

    # == init exp_dir ==
    ## define_experiment_workspace为实验创建一个独立的工作目录，并返回实验的名称和目录路径
    exp_name, exp_dir = define_experiment_workspace(cfg)
    ## 阻塞所有进程，直到所有进程都到达了这个同步点
    coordinator.block_all()
    ## 判断当前进程是否是主进程（master process），确保只有主进程执行创建目录或写入文件操作
    if coordinator.is_master():
        ## 在主进程中创建实验目录exp_dir，exist_ok=True确保如果目录已经存在，不会抛出异常
        os.makedirs(exp_dir, exist_ok=True)
        ## 将配置文件保存到实验目录中，cfg.to_dict()将配置对象转换为字典格式，以便保存
        save_training_config(cfg.to_dict(), exp_dir)
    ## 确保所有进程在主进程完成目录创建和配置保存后，继续执行后续操作
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    ## 创建一个日志记录器（logger），支持将日志同时输出到控制台（stdout）和exp_dir目录的日志文件
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    ## pformat用于“漂亮打印”（pretty-printing）复杂的数据结构
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        ## 创建一个TensorBoard的SummaryWriter，日志文件将保存在exp_dir目录下
        tb_writer = create_tensorboard_writer(exp_dir)
        ## 不启动wandb
        if cfg.get("wandb", False):  ## False
            ## WandB是一个强大的实验跟踪和可视化工具，支持更丰富的功能（如超参数搜索、模型版本管理等）
            ## 初始化WandB，设置项目名称、实验名称、配置和日志目录
            wandb.init(project="minisora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    ## 根据配置创建Colossal-AI的优化插件（如Zero2或Zero2-seq）
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),  ## 采用zero2插件
        dtype=cfg_dtype,  ## "bf16"
        grad_clip=cfg.get("grad_clip", 0),  ## 梯度裁剪的最大范数为1.0
        sp_size=cfg.get("sp_size", 1),  ## 序列并行的大小为1
    )
    ## Booster是Colossal-AI中的核心组件，用于封装和优化模型训练过程，支持多种并行策略和优化技术
    ## 通过插件（plugin）来配置具体的优化策略
    booster = Booster(plugin=plugin)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    ## cfg.dataset.type为“VideoTextDataset”
    assert cfg.dataset.type == "VideoTextDataset", "Only support VideoTextDataset for vae training"

    ## DATASETS是使用MMEngine的Registry类来创建一个注册表，用于管理和动态构建数据集类
    ## build_module函数，用于根据配置动态构建模块
    ## dataset为VideoTextDataset类型
    dataset = build_module(cfg.dataset, DATASETS)  ## 调用了build_from_cfg，返回与cfg.dataset.type为名的类的实例
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,  ## 数据集对象，用于加载数据
        batch_size=cfg.batch_size,  ## 1，每个批次的样本数量
        num_workers=cfg.get("num_workers", 4),  ## 16，数据加载器的子进程数量
        seed=cfg.get("seed", 1024),  ## 42，随机种子，用于数据采样的随机性
        shuffle=True,  ## 是否在每个epoch开始时打乱数据
        drop_last=True,  ## 是否丢弃最后一个不完整的批次
        pin_memory=True,  ## 是否将数据加载到GPU的pinned memory中，以加速数据传输（数据可以直接从CPU内存传输到GPU内存，而无需经过额外的拷贝步骤）
        process_group=get_data_parallel_group(),  ## 数据并行组，用于分布式训练，dist.group.WORLD
    )
    ## 根据 dataloader_args 配置创建数据加载器
    ## 返回两个对象：数据加载器DataloaderForVideo实例，采样器StatefulDistributedSampler实例
    dataloader, sampler = prepare_dataloader(**dataloader_args)
    ## 计算分布式训练中每个epoch的总批次大小
    ## cfg.batch_size：每个进程的本地批次大小，为1
    ## dist.get_world_size()：分布式环境中的总进程数
    ## cfg.get("sp_size", 1)：序列并行（Sequence Parallelism）的大小，默认为 1。
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.get("sp_size", 1)
    logger.info("Total batch size: %s", total_batch_size)
    ## 每个epoch的总步数，等于数据加载器的长度
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build vae model ==
    ## MODELS 是一个注册表实例，用于管理和动态构建模型
    ## locations=["opensora.models"]：指定注册表中模块的搜索路径
    ## 动态构建模型，并将其移动到指定设备上，同时将模型设置为训练模式
    ## model为VideoAutoencoderPipeline的实例
    model = build_module(cfg.model, MODELS).to(device, dtype).train()  ## 调用了build_from_cfg，返回与cfg.model.type为名的函数的返回
    ## 计算模型的总参数量（model_numel）和可训练参数量（model_numel_trainable）
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[VAE] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build discriminator model ==
    use_discriminator = cfg.get("discriminator", None) is not None  ## discriminator为None
    if use_discriminator:  # False
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype).train()
        discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
        logger.info(
            "[Discriminator] Trainable model params: %s, Total model params: %s",
            format_numel_str(discriminator_numel_trainable),
            format_numel_str(discriminator_numel),
        )

    # == setup loss functions ==
    vae_loss_fn = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        perceptual_loss_weight=cfg.get("perceptual_loss_weight", 0.1),
        kl_loss_weight=cfg.get("kl_loss_weight", 1e-6),
        device=device,
        dtype=dtype,
    )

    if use_discriminator:
        adversarial_loss_fn = AdversarialLoss(
            discriminator_factor=cfg.get("discriminator_factor", 1),
            discriminator_start=cfg.get("discriminator_start", -1),
            generator_factor=cfg.get("generator_factor", 0.5),
            generator_loss_type=cfg.get("generator_loss_type", "hinge"),
        )

        disc_loss_fn = DiscriminatorLoss(
            discriminator_factor=cfg.get("discriminator_factor", 1),
            discriminator_start=cfg.get("discriminator_start", -1),
            discriminator_loss_type=cfg.get("discriminator_loss_type", "hinge"),
            lecam_loss_weight=cfg.get("lecam_loss_weight", None),
            gradient_penalty_loss_weight=cfg.get("gradient_penalty_loss_weight", None),
        )

    # == setup vae optimizer ==
    ## HybridAdam是Adam优化器的一个变体，结合了CPU和GPU的优化能力，支持混合精度计算（如FP16和FP32），并针对大规模模型训练进行了优化
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),  ## 筛选出模型中需要梯度更新的参数
        adamw_mode=True,  ## 指定优化器是否使用AdamW模式
        lr=cfg.get("lr", 1e-5),  ## 学习率为1e-5
        weight_decay=cfg.get("weight_decay", 0),  ## weight_decay为0
    )
    lr_scheduler = None

    # == setup discriminator optimizer ==
    if use_discriminator:
        disc_optimizer = HybridAdam(
            filter(lambda p: p.requires_grad, discriminator.parameters()),
            adamw_mode=True,
            lr=cfg.get("lr", 1e-5),
            weight_decay=cfg.get("weight_decay", 0),
        )
        disc_lr_scheduler = None

    # == additional preparation ==
    ## 根据配置动态启用梯度检查点（Gradient Checkpointing）功能
    if cfg.get("grad_checkpoint", False):  ## True
        set_grad_checkpoint(model)  ## 开启梯度检查点，减少了激活值的存储需求，从而节省内存，用时间换空间

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)  ## bf16
    ## 将传入的模型、优化器、学习率调度器和数据加载器封装或替换为更高效的版本，以支持分布式训练、混合精度训练、内存优化、梯度累积、梯度检查点、其他优化
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    if use_discriminator:
        discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
            model=discriminator,
            optimizer=disc_optimizer,
            lr_scheduler=disc_lr_scheduler,
        )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)  ## 100
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = running_disc_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    ## 加载检查点以继续训练
    if cfg.get("load", None) is not None:  ## None
        logger.info("Loading checkpoint")
        start_epoch, start_step = load(
            booster,
            cfg.load,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=sampler,
        )
        if use_discriminator and os.path.exists(os.path.join(cfg.load, "discriminator")):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
        ## 用于同步所有进程，确保所有进程都完成加载操作后再继续执行。这可以避免因进程间不同步导致的潜在问题
        dist.barrier()
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    # =======================================================
    # 5. training loop
    # =======================================================
    ## 用于同步所有进程，确保所有进程都完成加载操作后再继续执行。这可以避免因进程间不同步导致的潜在问题
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):  ## range(0, 100)
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)  ## StatefulDistributedSampler
        dataiter = iter(dataloader)  ## DataloaderForVideo
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        ## tqdm 是一个流行的 Python 库，用于在终端或 Jupyter Notebook 中显示进度条，帮助用户直观地了解任务的执行进度
        with tqdm(
                enumerate(dataiter, start=start_step),  ## 为每个迭代项添加索引（从 start_step 开始）
                desc=f"Epoch {epoch}",  ## 设置进度条的描述信息，显示当前的 epoch 编号
                disable=not coordinator.is_master(),  ## 确保只有主进程（master）显示进度条，避免多个进程重复显示
                total=num_steps_per_epoch,  ## 进度条的总步数，表示每个 epoch 的总迭代次数
                initial=start_step,  ## 进度条的初始值，表示从 start_step 开始
        ) as pbar:
            for step, batch in pbar:
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

                # == mixed training setting ==
                mixed_strategy = cfg.get("mixed_strategy", None)  ## mixed_video_image
                if mixed_strategy == "mixed_video_image":
                    if random.random() < cfg.get("mixed_image_ratio", 0.0):  ## 0.2
                        x = x[:, :, :1, :, :]  ## 执行单帧图像的转换
                elif mixed_strategy == "mixed_video_random":
                    length = random.randint(1, x.size(2))
                    x = x[:, :, :length, :, :]  ## 从时间维度中截取前length帧

                # == vae encoding & decoding ===
                ## x --空间编码器--> x_z --时间编码器--> posterior --采样--> z --时间解码器--> x_z_rec --空间解码器--> x_rec
                x_rec, x_z_rec, z, posterior, x_z = model(x)  ## VideoAutoencoderPipeline

                # == loss initialization ==
                vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                disc_loss = torch.tensor(0.0, device=device, dtype=dtype)
                log_dict = {}

                # == loss: real image reconstruction ==
                ## 计算x和x_rec的重建损失，加权重建损失，加权KL损失
                ## 即时间+空间VAE的损失
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                log_dict["kl_loss"] = weighted_kl_loss.item()
                log_dict["nll_loss"] = weighted_nll_loss.item()
                if cfg.get("use_real_rec_loss", False):
                    vae_loss += weighted_nll_loss + weighted_kl_loss

                # == loss: temporal vae reconstruction ==
                ## 计算x_z和x_z_rec的加权重建损失
                ## 即时间VAE的损失
                _, weighted_z_nll_loss, _ = vae_loss_fn(x_z, x_z_rec, posterior, no_perceptual=True)
                log_dict["z_nll_loss"] = weighted_z_nll_loss.item()
                if cfg.get("use_z_rec_loss", False):
                    vae_loss += weighted_z_nll_loss

                # == loss: image only distillation ==
                ## 计算x_z和z的加权重建损失（不包含感知损失）
                ## 这里由于时间维度为1，因此理论上时间编码器
                if cfg.get("use_image_identity_loss", False) and x.size(2) == 1:  ## use_image_identity_loss: True
                    _, image_identity_loss, _ = vae_loss_fn(x_z, z, posterior, no_perceptual=True)
                    vae_loss += image_identity_loss
                    log_dict["image_identity_loss"] = image_identity_loss.item()

                # == loss: generator adversarial ==
                if use_discriminator:
                    recon_video = rearrange(x_rec, "b c t h w -> (b t) c h w").contiguous()
                    global_step = epoch * num_steps_per_epoch + step
                    fake_logits = discriminator(recon_video.contiguous())
                    adversarial_loss = adversarial_loss_fn(
                        fake_logits,
                        nll_loss,
                        model.module.get_temporal_last_layer(),
                        global_step,
                        is_training=model.training,
                    )
                    log_dict["adversarial_loss"] = adversarial_loss.item()
                    vae_loss += adversarial_loss

                # == generator backward & update ==
                optimizer.zero_grad()
                ## 使用 booster 工具进行反向传播，计算梯度
                booster.backward(loss=vae_loss, optimizer=optimizer)
                ## 根据计算得到的梯度更新模型参数
                optimizer.step()
                ## 将所有节点的损失值进行平均，确保全局损失的一致性
                all_reduce_mean(vae_loss)
                ## 将当前批次的损失值累加到运行损失（running_loss）中
                running_loss += vae_loss.item()

                # == loss: discriminator adversarial ==
                if use_discriminator:
                    real_video = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
                    fake_video = rearrange(x_rec, "b c t h w -> (b t) c h w").contiguous()
                    real_logits = discriminator(real_video.contiguous().detach())
                    fake_logits = discriminator(fake_video.contiguous().detach())
                    weighted_d_adversarial_loss, _, _ = disc_loss_fn(
                        real_logits,
                        fake_logits,
                        global_step,
                    )
                    disc_loss = weighted_d_adversarial_loss
                    log_dict["disc_loss"] = disc_loss.item()

                    # == discriminator backward & update ==
                    disc_optimizer.zero_grad()
                    booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                    disc_optimizer.step()
                    all_reduce_mean(disc_loss)
                    running_disc_loss += disc_loss.item()

                # == update log info ==
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    avg_disc_loss = running_disc_loss / log_step
                    # progress bar
                    pbar.set_postfix(
                        {"loss": avg_loss, "disc_loss": avg_disc_loss, "step": step, "global_step": global_step}
                    )
                    # tensorboard
                    tb_writer.add_scalar("loss", vae_loss.item(), global_step)
                    # wandb
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "num_samples": global_step * total_batch_size,
                                "epoch": epoch,
                                "loss": vae_loss.item(),
                                "avg_loss": avg_loss,
                                **log_dict,
                            },
                            step=global_step,
                        )
                    running_loss = running_disc_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    save(
                        booster,
                        exp_dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                        sampler=sampler,
                    )

                    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step + 1}")
                    if use_discriminator:
                        booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                        booster.save_optimizer(
                            disc_optimizer, os.path.join(save_dir, "disc_optimizer"), shard=True, size_per_shard=4096
                        )
                    dist.barrier()

                    logger.info(
                        "Saved checkpoint at epoch %s step %s global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        exp_dir,
                    )

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
