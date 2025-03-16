import collections
import functools
import queue
import random
import threading
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, _utils
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataloader import (
    IterDataPipe,
    MapDataPipe,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _sharding_worker_init_fn,
    _SingleProcessDataLoaderIter,
)

from .datasets import BatchFeatureDataset, VariableVideoTextDataset, VideoTextDataset
from .pin_memory_cache import PinMemoryCache
from .sampler import BatchDistributedSampler, StatefulDistributedSampler, VariableVideoBatchSampler


def _pin_memory_loop(
    in_queue, out_queue, device_id, done_event, device, pin_memory_cache: PinMemoryCache, pin_memory_key: str
):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                assert isinstance(data, dict)
                if pin_memory_key in data:
                    val = data[pin_memory_key]
                    pin_memory_value = pin_memory_cache.get(val)
                    pin_memory_value.copy_(val)
                    data[pin_memory_key] = pin_memory_value
            except Exception:
                data = ExceptionWrapper(where=f"in pin memory thread for device {device_id}")
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


class _MultiProcessingDataLoaderIterForVideo(_MultiProcessingDataLoaderIter):
    pin_memory_key: str = "video"

    def __init__(self, loader):
        _BaseDataLoaderIter.__init__(self, loader)
        self.pin_memory_cache = PinMemoryCache()

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank
            )

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                    self._shared_seed,
                ),
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
                custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
                current_device = custom_device_mod.current_device()
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    current_device,
                    self._pin_memory_thread_done_event,
                    self._pin_memory_device,
                    self.pin_memory_cache,
                    self.pin_memory_key,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue  # type: ignore[assignment]

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit

            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def remove_cache(self, output_tensor: torch.Tensor):
        self.pin_memory_cache.remove(output_tensor)

    def get_cache_info(self) -> str:
        return str(self.pin_memory_cache)


class DataloaderForVideo(DataLoader):
    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)  ## PyTorch 内置的单进程迭代器
        else:
            self.check_worker_number_rationality()  ## 确保 num_workers 的值是合理的（例如，不超过可用 CPU 核心数）
            return _MultiProcessingDataLoaderIterForVideo(self)  ## 自定义的多进程迭代器类


# Deterministic dataloader
## 为 PyTorch 的多进程数据加载器（DataLoader）生成一个种子设置函数，用于确保每个数据加载子进程（worker）的随机性是可复现的
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader(
    dataset,  ## e.g. VideoTextDataset的实例，表示要加载的数据集
    batch_size=None,  ## 指定每个批次加载的数据样本数量
    shuffle=False,  ## 是否在每个 epoch 开始时随机打乱数据集
    seed=1024,
    drop_last=False,  ## 在最后一个批次中，如果数据集的大小不能被 batch_size 整除，是否丢弃最后一个不完整的批次
    pin_memory=False,  ## 是否将数据加载到 GPU 的固定内存中
    num_workers=0,  ## 指定加载数据时使用的子进程数量，如果为 0，则在主进程中加载数据；如果大于 0，则使用多进程加载数据
    process_group: Optional[ProcessGroup] = None,  ## 在分布式训练中，指定进程组
    bucket_config=None,  ## 用于配置数据分桶策略，分桶通常用于处理变长数据（如视频帧数不同），通过将相似长度的数据分到同一个桶中，减少填充（padding）带来的计算开销
    num_bucket_build_workers=1,  ## 在构建分桶时使用的子进程数量
    prefetch_factor=None,  ## 指定每个子进程预取的数据批次数量
    cache_pin_memory=False,  ## 是否将缓存数据固定在内存中，如果为 True，可以加速数据加载速度，但会占用更多内存
    **kwargs,  ## 用于传递额外的关键字参数，可以被 torch.utils.data.DataLoader 或其他相关函数使用
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, VariableVideoTextDataset):
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
        )
        dl_cls = DataloaderForVideo if cache_pin_memory else DataLoader
        return (
            dl_cls(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VideoTextDataset):
        process_group = process_group or _get_default_group()
        ## 用于分布式训练的采样器，负责在多个进程（或设备）之间分配数据集的样本
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),  ## 根据进程组的大小来确定数据集的分割份数
            rank=process_group.rank(),  ## 当前进程的编号
            shuffle=shuffle,
        )
        ## 动态选择数据加载器（DataLoader）的类
        dl_cls = DataloaderForVideo if cache_pin_memory else DataLoader
        return (
            dl_cls(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),  ## 每个子进程（worker）初始化时调用的函数
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,  ## 定义如何将多个样本组合成一个批次
                prefetch_factor=prefetch_factor,  ## 指定每个子进程预取的数据批次数量
                **_kwargs,
            ),
            sampler,
        )
    elif isinstance(dataset, BatchFeatureDataset):
        sampler = BatchDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
        )
        return (
            DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_batch,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def collate_fn_default(batch):
    # filter out None
    batch = [x for x in batch if x is not None]

    # HACK: for loading text features
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]

        texts = [x.pop("text") for x in batch]
        texts = torch.cat(texts, dim=1)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["text"] = texts
    return ret


def collate_fn_batch(batch):
    """
    Used only with BatchDistributedSampler
    """
    # filter out None
    batch = [x for x in batch if x is not None]

    res = torch.utils.data.default_collate(batch)

    # squeeze the first dimension, which is due to torch.stack() in default_collate()
    if isinstance(res, collections.abc.Mapping):
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.squeeze(0)
    elif isinstance(res, collections.abc.Sequence):
        res = [x.squeeze(0) if isinstance(x, torch.Tensor) else x for x in res]
    elif isinstance(res, torch.Tensor):
        res = res.squeeze(0)
    else:
        raise TypeError

    return res
