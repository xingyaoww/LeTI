import math
import jax.numpy as jnp
import numpy as np
import threading
from datasets import Dataset
from queue import Queue
from typing import Deque, List, Callable, Iterable, Optional, Tuple

from leti.utils.training import pad_to_batch_size

class DataloaderPrefetchWrapper:
    def __init__(
        self, 
        dataloader: Iterable,
        prefetch_size: int = 4,
    ):
        """
        # some testcases
        fake_dl = list(range(6))
        dl_wrapped = DataloaderPrefetchWrapper(iter(fake_dl), prefetch_size=8)

        outputs = []
        for i in dl_wrapped:
            outputs.append(i)
        assert outputs == fake_dl
        """
        self.dataloader: Iterable = dataloader
        self.buffer_size: int = prefetch_size
        
        self.data_queue: Queue = Queue(maxsize=self.buffer_size)
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_thread_fn,
            daemon=True,
        )
        self.prefetch_thread.start()

    def _prefetch_thread_fn(self):
        while True:
            # load next element (in background)
            try:
                next_element = next(self.dataloader)
            except StopIteration:
                next_element = None
            # it will block if the queue is full
            self.data_queue.put(next_element)

            if next_element is None:
                break


    def __iter__(self):
        return self

    def __next__(self):
        # get next element
        next_element = self.data_queue.get()
        if next_element is None:
            raise StopIteration
        return next_element

def dataloader_impl(
    dataset: Dataset,
    batch_size: int,
    return_idx: bool = False,
    return_jnp_array: bool = False,
):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    # require shuffle to be done in dataset
    batch_idx = np.arange(len(dataset))

    # dataset should pad the last batch to batch_size
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {
            k: jnp.array(v) if return_jnp_array \
                else np.array(v)
            for k, v in batch.items()
        }
        if return_idx:
            yield idx, batch
        else:
            yield batch

def dataloader(*args, **kwargs):
    """
    This is a wrapper around the dataloader_impl that prefetches the next batch.
    """
    return DataloaderPrefetchWrapper(dataloader_impl(*args, **kwargs))

def generation_dataloader_impl(
    dataset,
    n_samples: int,
    batch_size: int,
    tokenizer,
    pad_to_multiple_of: int = 128,
    return_jnp_array: bool = False,
    jnp_keys = ["input_ids", "attention_mask"],
):
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            # return_tensors="jax",
            padding=True,
            # use a large pad_to_multiple_of to avoid repeated jax compilation
            pad_to_multiple_of=pad_to_multiple_of
        ),
    )
    # calculate length of input_ids and sort by it
    dataset = dataset.map(
        lambda x: {"seq_length": len(x["input_ids"])}
    )
    # sort by seq_length AND id (to keep the order of the same length prompts)
    dataset = dataset.sort(["seq_length", "id"])

    def element_iterator():
        for row in dataset:
            # yield n_samples of the same prompt
            for generation_idx in range(n_samples):
                yield {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],

                    # extra fields
                    "text": row["text"],
                    "reference": row["reference"],
                    "prompt_idx": row["id"],
                    "generation_idx": generation_idx,
                    "seq_length": row["seq_length"],
                }

    def process_cur_batch(cur_batch, batch_size, tokenizer):
        cur_batch_dict = {
            key: np.array([element[key] for element in cur_batch])
            for key in cur_batch[0].keys()
        }

        ret = pad_to_batch_size(
            cur_batch_dict,
            num_examples=len(cur_batch),
            batch_size=batch_size,
            tokenizer=tokenizer,
            ignore_keys=["reference", "prompt_idx", "generation_idx", "seq_length", "text"],
        )

        if return_jnp_array:
            ret = {
                k: jnp.array(v) if k in jnp_keys else v
                for k, v in ret.items()
            }
        return ret

    cur_seq_length = None
    cur_batch = []
    for element in element_iterator():
        if cur_seq_length is None:
            cur_seq_length = element["seq_length"]

        if len(cur_batch) == batch_size or cur_seq_length != element["seq_length"]:
            # we only want to yield batches of elements of the same length

            yield process_cur_batch(cur_batch, batch_size, tokenizer)

            # update for new batch
            cur_batch = []
            cur_seq_length = element["seq_length"]
        
        cur_batch.append(element)

    # return last batch
    if len(cur_batch) > 0:
        yield process_cur_batch(cur_batch, batch_size, tokenizer)

def generation_dataloader(*args, **kwargs):
    """
    This is a wrapper around the generation_dataloader_impl that prefetches the next batch.
    """
    return DataloaderPrefetchWrapper(generation_dataloader_impl(*args, **kwargs))
