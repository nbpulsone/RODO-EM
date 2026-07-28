import torch
from torch.utils import data
from transformers import AutoTokenizer
from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None,
                 hp=None,
                 dom_ids=None,
                 cc_partition=False):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.domain_labels = dom_ids
        self.cc_labels = []
        self.max_len = max_len
        self.size = size
        self.hp = hp
        
        # lines associated with this dataset
        if isinstance(path, list):
            lines = path
        else:
            lines = list(open(path))

        if cc_partition:
            for line in lines:
                s1, s2, label, cc_label = line.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))

                # accept 0/1 as well as true/false strings.
                self.cc_labels.append(int(str(cc_label).strip().lower() in {"1", "true"}))
        else:
            for line in lines:
                s1, s2, label = line.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        if self.domain_labels is not None:
            self.domain_labels = self.domain_labels[:size]
        if len(self.cc_labels) > 0:
            self.cc_labels = self.cc_labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)

        # augment if da is set, provide domain label if available
        # ugly but workable
        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(
                left + ' [SEP] ' + right,
                self.da
            )
            left, right = combined.split(' [SEP] ')

            x_aug = self.tokenizer.encode(
                text=left,
                text_pair=right,
                max_length=self.max_len,
                truncation=True
            )

            if self.domain_labels is not None and len(self.cc_labels) > 0:
                return (
                    x,
                    x_aug,
                    self.labels[idx],
                    self.domain_labels[idx],
                    self.cc_labels[idx]
                )
            elif self.domain_labels is not None:
                return x, x_aug, self.labels[idx], self.domain_labels[idx]
            elif len(self.cc_labels) > 0:
                return x, x_aug, self.labels[idx], self.cc_labels[idx]
            else:
                return x, x_aug, self.labels[idx]

        else:
            if self.domain_labels is not None and len(self.cc_labels) > 0:
                return (
                    x,
                    self.labels[idx],
                    self.domain_labels[idx],
                    self.cc_labels[idx]
                )
            elif self.domain_labels is not None:
                return x, self.labels[idx], self.domain_labels[idx]
            elif len(self.cc_labels) > 0:
                return x, self.labels[idx], self.cc_labels[idx]
            else:
                return x, self.labels[idx]

    @staticmethod
    def pad(batch):
        # Standard input with domain and CC labels:
        # (x, y, domain_id, cc_label)
        if len(batch[0]) == 4 and isinstance(batch[0][1], int):
            x12, y, d, cc = zip(*batch)

            maxlen = max(len(x) for x in x12)
            x12 = [xi + [0] * (maxlen - len(xi)) for xi in x12]

            return (
                torch.LongTensor(x12),
                torch.LongTensor(y),
                torch.LongTensor(d),
                torch.LongTensor(cc)
            )

        # Augmented input with domain and CC labels:
        # (x, x_aug, y, domain_id, cc_label)
        elif len(batch[0]) == 5:
            x1, x2, y, d, cc = zip(*batch)

            maxlen = max(len(x) for x in x1 + x2)
            x1 = [xi + [0] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen - len(xi)) for xi in x2]

            return (
                torch.LongTensor(x1),
                torch.LongTensor(x2),
                torch.LongTensor(y),
                torch.LongTensor(d),
                torch.LongTensor(cc)
            )
        elif len(batch[0]) == 3 and isinstance(batch[0][1], int):
            x12, y, d = zip(*batch)
            maxlen = max(len(x) for x in x12)
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), torch.LongTensor(y), torch.LongTensor(d)
        elif len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)
