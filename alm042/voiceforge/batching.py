
import random
from collections.abc import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler


class LengthBucketSampler(Sampler[list[int]]):
    """
    Build batches out of samples with similar lengths.

    A batch is padded to its longest sample, so pairing a 600-frame clip with
    150-frame clips wastes most of the compute in that batch.  Sorting by
    length and shuffling inside a buffer of several batches removes most of
    that padding waste while keeping batch composition varied between epochs.

    ``lengths[i]`` is the Mel length of dataset sample ``i`` (for example
    ``duration * 22050 / 256``).
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        buffer_batches: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.buffer_batches = max(1, int(buffer_batches))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        total = len(self.lengths)

        if self.drop_last:
            return total // self.batch_size

        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        order = sorted(
            range(len(self.lengths)),
            key=lambda index: self.lengths[index],
        )

        group_size = self.batch_size * self.buffer_batches
        batches: list[list[int]] = []

        for start in range(0, len(order), group_size):
            chunk = order[start:start + group_size]

            if self.shuffle:
                rng.shuffle(chunk)

            for offset in range(0, len(chunk), self.batch_size):
                batch = chunk[offset:offset + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches


def collate_batch(samples):
    """
    Combine dataset samples into one padded training batch.

    Supports:
      - Dictionary samples using text_sequence and mel.
      - Dictionary samples using tensor-valued text and mel.
      - Tuple samples containing (text, mel).
    """

    texts = []
    mels = []

    for sample in samples:
        if isinstance(sample, dict):
            # LJSpeechDataset stores the original sentence in "text".
            # The model needs the numerical sequence instead.
            if "text_sequence" in sample:
                text = sample["text_sequence"]
            else:
                text = sample["text"]

            mel = sample["mel"]

        else:
            text, mel = sample

        text = torch.as_tensor(text, dtype=torch.long)
        mel = torch.as_tensor(mel, dtype=torch.float32)

        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)

        if text.ndim != 1:
            raise ValueError("Each text sequence must be one-dimensional")

        if mel.ndim != 2:
            raise ValueError("Each Mel spectrogram must be two-dimensional")

        if text.numel() == 0 or mel.shape[1] == 0:
            raise ValueError("Text and Mel sequences cannot be empty")

        texts.append(text)
        mels.append(mel)

    text_lengths = torch.tensor(
        [text.numel() for text in texts],
        dtype=torch.long,
    )

    mel_lengths = torch.tensor(
        [mel.shape[1] for mel in mels],
        dtype=torch.long,
    )

    padded_texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=0,
    )

    padded_mels = pad_sequence(
        [mel.transpose(0, 1) for mel in mels],
        batch_first=True,
        padding_value=0.0,
    ).transpose(1, 2)

    return {
        "text": padded_texts,
        "text_lengths": text_lengths,
        "mel": padded_mels,
        "mel_lengths": mel_lengths,
    }
