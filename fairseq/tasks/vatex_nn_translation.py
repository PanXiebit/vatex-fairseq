
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import itertools
from fairseq.utils import new_arange
from fairseq.tasks.translation import TranslationTask
from fairseq import utils
from fairseq import metrics, options, utils
import logging
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.vatex_nn_tgt_datasets import VatexNNTgtDataset


from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
)

logger = logging.getLogger(__name__)


def load_videolang_dataset(data_path, split,
    src, tgt, tgt_dict, combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
    append_eos_to_target, append_bos_to_target):
    """Load a given dataset split.
    Args:
        split (str): name of the split (e.g., train, valid, test)
    """

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)


    def vatex_indexed_dataset(path):
        if dataset_impl == "raw":
            return indexed_dataset.VatexIndexedRawTextDataset(path)
        else:
            raise NotImplementedError("the dataset_impl of video language dataset must be raw.")

    def text_indexed_dataset(path, tgt_dict):
        if dataset_impl == "raw":
            return indexed_dataset.IndexedRawNNTextDataset(path, tgt_dict)
        else:
            raise NotImplementedError("the dataset_impl of video language dataset must be raw.")

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(vatex_indexed_dataset(prefix + src))
        tgt_datasets.append(text_indexed_dataset(prefix + tgt, tgt_dict))

        print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

        if not combine:
            break
    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return VatexNNTgtDataset(
        src_dataset, src_dataset.sizes,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        shuffle=False, input_feeding=True,
        append_eos_to_target=append_eos_to_target,
        append_bos_to_target=append_bos_to_target,
    )


@register_task('vatex_nn_translation')
class VatexNNTranslationTask(FairseqTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument(
            '--append_eos_to_target', default=True, type=bool,
            help="the path of the mean file.")
        parser.add_argument(
            '--append_bos_to_target', default=True, type=bool,
            help="the path of the mean file.")

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        return cls(args, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_videolang_dataset(
            data_path, split, src, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            append_eos_to_target=self.args.append_eos_to_target,
            append_bos_to_target=self.args.append_bos_to_target
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, args):
        from fairseq.iterative_refinement_vatex_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        sample['net_input']['prev_nn_tokens'] = self.inject_noise(sample['net_input']['prev_nn_tokens'])
        sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'][:, 1:]  # remove eos
        sample['target'] = sample['target'][:, 1:]   # remove bos

        # print(sample.keys(), sample["net_input"].keys())
        # print(sample["nn_tokens"][:5], sample['net_input']['prev_nn_tokens'][:5])
        # print(sample["target"][:5], sample['net_input']['prev_output_tokens'][:5])
        # print(sample['tgt_lengths'][:5], sample['nn_lengths'][:5])
        # tgt_sent = sample["target"][0]
        # nn_tokens = sample["nn_tokens"][0]
        # print(self.tgt_dict.string(tgt_sent))
        # print(self.tgt_dict.string(nn_tokens))
        # exit()

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['net_input']['prev_nn_tokens'] = self.inject_noise(sample['net_input']['prev_nn_tokens'])
            sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'][:, 1:]  # remove eos
            sample['target'] = sample['target'][:, 1:]  # remove bos
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict


    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)
