# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import os, random
from . import data_utils, FairseqDataset

logger = logging.getLogger(__name__)

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if len(values[0].shape) > 1:   # the shape of value is [1, num_frames, 1024]
        size = max(v.size(1) for v in values)  # num_frames of source video
        res = values[0].new(len(values), size, values[0].size(2)).fill_(0)  # video pad with zero
        for i, v in enumerate(values):
            v = v.squeeze()
            copy_tensor(v, res[i][size - v.size(0):] if left_pad else res[i][:v.size(0)])
    else:
        size = max(v.size(0) for v in values)  # tgt_len of target
        res = values[0].new(len(values), size).fill_(pad_idx)   # sentence pad with pad_idx
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate(
    samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}
    def merge(key, left_pad, pad_idx, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_videos = merge('source', left_pad=False, pad_idx=0)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].size(1) for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_videos = src_videos.index_select(0, sort_order)

    prev_output_tokens, prev_nn_tokens = None, None
    target, tgt_lengths, nn_tokens, nn_lengths = None, None, None, None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target, pad_idx=pad_idx)
        nn_tokens = merge("nn_tokens", left_pad=False, pad_idx=pad_idx)
        target = target.index_select(0, sort_order)
        nn_tokens = nn_tokens.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        nn_lengths = torch.LongTensor([s['nn_tokens'].numel() for s in samples]).index_select(0, sort_order)
        tgt_lengths = tgt_lengths.index_select(0, sort_order)
        nn_lengths = nn_lengths.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # print("Input feeding is True!")
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,  # TODO?, easy for right_shift
                pad_idx=pad_idx
            )
            prev_nn_tokens = merge(
                'nn_tokens',
                left_pad=False,
                move_eos_to_beginning=False,
                pad_idx=pad_idx
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            prev_nn_tokens = prev_nn_tokens.index_select(0, sort_order)
    else:
        ntokens = None

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_videos': src_videos,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nn_tokens': nn_tokens,
        'tgt_lengths': tgt_lengths,
        'nn_lengths': nn_lengths,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['prev_nn_tokens'] = prev_nn_tokens
    return batch


class VatexNNTgtDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, append_eos_to_target=False,
        append_bos_to_target=False
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target  # True
        self.append_bos_to_target = append_bos_to_target  # True

    def __getitem__(self, index):
        tgt_item = self.tgt.get_tgt_tokens(index) if self.tgt is not None else None
        nn_item = self.tgt.get_nn_tokens(index) if self.tgt is not None else None
        src_item = torch.FloatTensor(np.load(self.src[index]))
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos()
            if self.tgt and tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        if self.append_bos_to_target:
            bos = self.tgt_dict.bos()
            if self.tgt and tgt_item[0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), tgt_item])


        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'nn_tokens': nn_item,
        }

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.tgt_dict.pad(), eos_idx=self.tgt_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # if self.tgt_sizes is not None:
        #     indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
