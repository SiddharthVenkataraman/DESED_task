from torch.utils.data import Sampler
import numpy as np
import lightning_fabric.utilities.distributed
import torch

class ConcatDatasetBatchSampler(Sampler):
    """This sampler is built to work with a standard Pytorch ConcatDataset.
    From SpeechBrain dataio see https://github.com/speechbrain/

    It is used to retrieve elements from the different concatenated datasets placing them in the same batch
    with proportion specified by batch_sizes, e.g 8, 16 means each batch will
    be of 24 elements with the first 8 belonging to the first dataset in ConcatDataset
    object and the last 16 to the second.
    More than two datasets are supported, in that case you need to provide 3 batch
    sizes.

    Note
    ----
    Batched are drawn from the datasets till the one with smallest length is exhausted.
    Thus number of examples in your training epoch is dictated by the dataset
    whose length is the smallest.


    Arguments
    ---------
    samplers : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    batch_sizes: list
        Batch sizes.
    epoch : int
        The epoch to start at.
    """

    def __init__(self, samplers, batch_size: (tuple, list), epoch=0, drop_last=False) -> None:  # Update. Renames batch_sizes to batch_size, and added drop_last

        # Major updates to ConcatDatasetBatchSampler, in order to allow for lightning_fabric's new kind of Data sampler
        # Current version checks if samplers is of type:
        #      (old) [list of 'torch.utils.data.sampler.RandomSampler']
        #   or (new) lightning_fabric.utilities.distributed.DistributedSamplerWrapper
        # and accordingly handles the in-built functions __init__(), __iter__(), __len__(), and set_epoch()

        batch_sizes = batch_size  # Update. Copied input batch_size to batch_sizes

        if isinstance(samplers, lightning_fabric.utilities.distributed.DistributedSamplerWrapper):  # Update. Added check to allow for DistributedSamplerWrapper
            # Current sampler is of lightning_fabric class
            if not isinstance(batch_sizes, (list, tuple)):
                raise ValueError(
                    "batch_sizes should be a list or tuple of integers, "
                    "but got batch_sizes={}".format(batch_sizes)
                )

            if not len(batch_sizes) == len(samplers.dataset._sampler.data_source.datasets):
                raise ValueError("batch_sizes and samplers should be have same length")

            self.batch_sizes = batch_sizes
            self.samplers = samplers
            self.offsets = [0] + samplers.dataset._sampler.data_source.cumulative_sizes  # Offsets to the first index of the 1st, 2nd, 3rd, ... dataset in the concatenated dataset

            self.epoch = epoch
            self.set_epoch(self.epoch)
            self.batch_size = batch_sizes  # Updated. Added self.batch_size. This variable seems to be called by lightning_ functions?
            self.drop_last = drop_last  # Updated. Added self.drop_last. Necessary?

            # Create random iterators for the different datasets. Used in the __iter__() function.
            # These iterators are a duplicate of those used by the original code (i.e., using class torch.utils.data.RandomSampler)
            self.backup_iterators = [torch.utils.data.RandomSampler(x) for x in self.samplers.dataset._sampler.data_source.datasets]

        else:
            if not isinstance(samplers, (list, tuple)):
                raise ValueError(
                    "samplers should be a list or tuple of Pytorch Samplers, "
                    "but got samplers={}".format(samplers)
                )

            if not isinstance(batch_sizes, (list, tuple)):
                raise ValueError(
                    "batch_sizes should be a list or tuple of integers, "
                    "but got batch_sizes={}".format(batch_sizes)
                )

            if not len(batch_sizes) == len(samplers):
                raise ValueError("batch_sizes and samplers should be have same length")

            self.batch_sizes = batch_sizes
            self.samplers = samplers
            self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

            self.epoch = epoch
            self.set_epoch(self.epoch)
            self.batch_size = batch_sizes  # Updated. Added self.batch_size
            self.drop_last = drop_last  # Updated. Added self.drop_last

            self.backup_iterators = None  # Updated. Added to be used when DistributedSamplerWrapper b

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if isinstance(self.samplers, lightning_fabric.utilities.distributed.DistributedSamplerWrapper):  # Find the "epoch" variable for the DistributedSamplerWrapper
            if hasattr(self, "epoch"):
                self.epoch = epoch
        else:
            if hasattr(self.samplers[0], "epoch"):
                for s in self.samplers:
                    s.set_epoch(epoch)

    def __iter__(self):

        if isinstance(self.samplers, lightning_fabric.utilities.distributed.DistributedSamplerWrapper):  # Update. Added check to allow for DistributedSamplerWrapper

            # Iterate with samplers of type DistributedSamplerWrapper.
            # Output of each iteration should be a list of indices.
            # If batch_sizes = [2,2,4],
            # Output = [x1, x2, y1, y2, z1, z2, z3, z4]
            #   Where x1, x2 are indices from the first dataset, and similarly for y's and the z's
            #   Each dataset has an offset, so x_min < x's < x_max. x_min and x_max can be found using the offsets[], and the batch_sizes[]

            iterators = [iter(i) for i in self.backup_iterators]
            tot_batch = []

            for b_num in range(len(self)):
                for samp_idx in range(len(self.batch_size)):
                    c_batch = []
                    while len(c_batch) < self.batch_size[samp_idx]:
                        c_batch.append(next(iterators[samp_idx]) + self.offsets[samp_idx])
                    tot_batch.extend(c_batch)
                yield tot_batch
                tot_batch = []

        else:
            iterators = [iter(i) for i in self.samplers]
            tot_batch = []

            for b_num in range(len(self)):
                for samp_idx in range(len(self.samplers)):
                    c_batch = []
                    while len(c_batch) < self.batch_sizes[samp_idx]:
                        c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                    tot_batch.extend(c_batch)

                yield tot_batch
                tot_batch = []

    def __len__(self):

        if isinstance(self.samplers, lightning_fabric.utilities.distributed.DistributedSamplerWrapper):  # Update. Added check to allow for DistributedSamplerWrapper
            # Finding length, i.e., number of batches, for DistributedSamplerWrapper
            # Length is defined as the maximum number of batches possible.
            # If (len(dataset_X) = 1000, batchsize_X = 10), (len(dataset_Y) = 200, batchsize_Y = 2),
            #   there can be 500 batches with X, but only 100 batches with Y.
            #   So, the length of this sampler is min(500,100) = Y, making data in Y the limiting factor.

            cum_sizes = self.samplers.dataset._sampler.data_source.cumulative_sizes
            ind_sizes = [cum_sizes[0]] + [y-x for (x, y) in zip(cum_sizes[0:-1], cum_sizes[1:])]

            c_lens = [int(x/y) for (x, y) in zip(ind_sizes, self.batch_sizes)]
            min_len = min(c_lens)

            return min_len

        else:
            min_len = float("inf")
            for idx, sampler in enumerate(self.samplers):
                c_len = (len(sampler)) // self.batch_sizes[idx]

                min_len = min(c_len, min_len)

        return min_len
