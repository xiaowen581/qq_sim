import torch
from torch import Tensor
from torch import cat, tensor, as_tensor, zeros, ones, long
from nlp_utils.file import read_file_as_str
from tqdm import tqdm
from transformers.models.bert import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, RandomSampler
from torch import LongTensor
from torch.utils.data import random_split, Dataset
from functools import wraps
from transformers import DataCollatorForLanguageModeling
from itertools import chain
from transformers import LineByLineTextDataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils import PreTrainedTokenizer
import random


def get_same_meaning_sentence(data):
    def get_sentence_by_id(data_pair_list, sentence_id):
        return data_pair_list[int(sentence_id / 2)][int(sentence_id % 2)]

    def find_root_id(sent_label_list: list, index: int):
        if sent_label_list[index] == index:
            return index

        sent_label_list[index] = find_root_id(sent_label_list, sent_label_list[index])
        return sent_label_list[index]

    # data = prepare_qqsim_data(train_file_v2)
    data_label_1 = [(sent_a, sent_b) for (sent_a, sent_b, label) in zip(*data) if label == 1]

    sent_label = [i for i in range(len(data_label_1) * 2)]
    sent_hash = {}
    for index, (sent_a, sent_b) in enumerate(data_label_1):
        index_a = index * 2
        index_b = index * 2 + 1

        index_root_a = index_a
        if sent_a in sent_hash.keys():
            index_root_a = find_root_id(sent_label, sent_hash[sent_a])
        else:
            sent_hash[sent_a] = index_a

        index_root_b = index_b
        if sent_b in sent_hash.keys():
            index_root_b = find_root_id(sent_label, sent_hash[sent_b])
        else:
            sent_hash[sent_b] = index_b

        sent_label[index_root_b] = index_root_a
        # 8 7422
    # 47268 47269
    for index in range(len(sent_label)):
        sent_label[index] = find_root_id(sent_label, sent_label[index])

    result = {}
    for index in range(len(sent_label)):
        if sent_label[index] in result.keys():
            result[sent_label[index]].append(index)
        else:
            result[sent_label[index]] = [index]

    same_sentences_list = []
    for key, value in result.items():
        if len(value) > 2:
            same_sentence = [get_sentence_by_id(data_label_1, index) for index in value]
            same_sentences_list.append(same_sentence)

    return same_sentences_list


def create_label_1_enhance_data(data):
    enhance_a = []
    enhance_b = []
    enhance_label = []
    same_sentences_list = get_same_meaning_sentence(data)
    for same_sentences in same_sentences_list:
        for i in range(len(same_sentences)):
            for j in range(i+1, len(same_sentences)):
                enhance_a.append(same_sentences[i])
                enhance_b.append(same_sentences[j])
                enhance_label.append(int(1))

                for _ in range(2):
                    enhance_a.append(same_sentences[i])
                    sentence_label_0 = data[random.randint(0, 1)][random.randint(0, len(data[0])-1)]
                    while sentence_label_0 in same_sentences_list:
                        sentence_label_0 = data[random.randint(0, 1)][random.randint(0, len(data[0]) - 1)]
                    enhance_b.append(sentence_label_0)
                    enhance_label.append(int(0))

    return enhance_a, enhance_b, enhance_label

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号


def get_token(tokenizer, content_a, content_b=None, pad_size=64):
    token = tokenizer.tokenize(content_a)
    token = [CLS] + token
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size

    if content_b:
        token_b = tokenizer.tokenize(content_b)
        token_b = [SEP] + token_b # + [SEP]
        mask_b = []
        token_ids_b = tokenizer.convert_tokens_to_ids(token_b)

        if pad_size:
            if len(token_b) < pad_size:
                mask_b = [1] * len(token_ids_b) + [0] * (pad_size - len(token_b))
                token_ids_b += ([0] * (pad_size - len(token_b)))
            else:
                mask_b = [1] * pad_size
                token_ids_b = token_ids_b[:pad_size]

    if content_b:
        return token_ids + token_ids_b, mask + mask_b
    else:
        return token_ids, mask


class OrgDataSet(Dataset):
    def __init__(self, *data):
        assert all(len(data[0]) == len(item) for item in data)
        self.data = data

    def __getitem__(self, index):
        return tuple(item[index] for item in self.data)

    def __len__(self):
        return len(self.data[0])

    def get_column(self, column_num):
        if column_num >= len(self.data):
            raise IndexError('column_num out of range')
        return self.data[column_num]


def prepare_qqsim_data(path: str, is_train=False):
    data = read_file_as_str(path)
    sentence_a = []
    sentence_b = []
    labels = []
    for index, line in enumerate(data):
        info = line.split('\t')
        if len(info) == 3:
            if len(info[0]) < 1 or len(info[1]) < 1:
                print('warning: data format error at line {}'.format(index + 1))
                continue
            sentence_a.append(info[0].strip())
            sentence_b.append(info[1].strip())
            labels.append(int(info[2]))
        elif len(info) == 2 and is_train is False:
            sentence_a.append(info[0].strip())
            sentence_b.append(info[1].strip())
            labels.append('')
        else:
            print('warning: data format error at line {}'.format(index + 1))
            continue
    return sentence_a, sentence_b, labels


def prepare_qqsim_train_test_org_data_loader(path: str, tokenizer, batch_size=64, test_scale=0.2):
    org_a, org_b, org_labels = prepare_qqsim_data(path, is_train=True)

    test_len = int(len(org_a) * test_scale)
    train_len = len(org_a) - test_len

    train_org_a, train_org_b, train_org_labels = org_a[:train_len], org_b[:train_len], org_labels[:train_len]
    test_a, test_b, test_labels = org_a[train_len:], org_b[train_len:], org_labels[train_len:]

    # 数据增强
    enhance_a, enhance_b, enhance_label = create_label_1_enhance_data((train_org_a, train_org_b, train_org_labels))
    train_org_a += enhance_a
    train_org_b += enhance_b
    train_org_labels += enhance_label

    data_a = train_org_a + train_org_b
    data_b = train_org_b + train_org_a
    data_labels = train_org_labels + train_org_labels

    train_data_set = OrgDataSet(data_a, data_b, data_labels)
    test_data_set = OrgDataSet(test_a, test_b, test_labels)

    num_0 = num_1 = 0
    for label in data_labels:
        if label is 0:
            num_0+=1
        else:
            num_1+=1

    print('num_0:{}, num_1:{}'.format(num_0, num_1))

    class CollateBatch:
        def __init__(self, token_text_a, token_text_b, text_a_len, text_b_len, label_list, input_ids, token_type_ids, attention_mask):
            self.batch_token_text_a = token_text_a
            self.batch_token_text_b = token_text_b
            self.batch_text_a_len = text_a_len
            self.batch_text_b_len = text_b_len
            self.batch_input_ids = input_ids
            self.batch_token_type_ids = token_type_ids
            self.batch_attention_mask = attention_mask
            self.batch_label_list = label_list

        def pin_memory(self):
            self.batch_token_text_a = self.batch_token_text_a.pin_memory()
            self.batch_token_text_b = self.batch_token_text_b.pin_memory()
            self.batch_text_a_len = self.batch_text_a_len.pin_memory()
            self.batch_text_b_len = self.batch_text_b_len.pin_memory()
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            return self

    def collate_wrapper(batch):
        text_a_list = []
        text_b_list = []
        text_a_len = []
        text_b_len = []
        text_a_b_list = []
        text_a_max_length = 0
        text_b_max_length = 0
        for batch_data in batch:
            text_a_b_list.append((batch_data[0], batch_data[1]))
            text_a_list.append(batch_data[0])
            text_b_list.append(batch_data[1])

            text_a_word_list = batch_data[0].split()
            text_b_word_list = batch_data[1].split()

            text_a_len.append(len(text_a_word_list))
            text_b_len.append(len(text_b_word_list))

            text_a_max_length = len(text_a_word_list) if len(text_a_word_list) > text_a_max_length else text_a_max_length
            text_b_max_length = len(text_b_word_list) if len(text_b_word_list) > text_b_max_length else text_b_max_length

        label_list = LongTensor([batch_data[2] for batch_data in batch])
        
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, label_list, **ret)

    # TODO: to support num_workers
    train_loader = DataLoader(train_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=True)#, num_workers=4)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=False)#, num_workers=4)

    return train_loader, test_loader



def prepare_qqsim_predict_org_data_loader(path: str, tokenizer, batch_size=1):
    org_data = prepare_qqsim_data(path)
    data_set = OrgDataSet(*org_data)

    class CollateBatch:
        def __init__(self, token_text_a, token_text_b, text_a_len, text_b_len, label_list, input_ids, token_type_ids, attention_mask):
            self.batch_token_text_a = token_text_a
            self.batch_token_text_b = token_text_b
            self.batch_text_a_len = text_a_len
            self.batch_text_b_len = text_b_len
            self.batch_input_ids = input_ids
            self.batch_token_type_ids = token_type_ids
            self.batch_attention_mask = attention_mask
            self.batch_label_list = label_list

        def pin_memory(self):
            self.batch_token_text_a = self.batch_token_text_a.pin_memory()
            self.batch_token_text_b = self.batch_token_text_b.pin_memory()
            self.batch_text_a_len = self.batch_text_a_len.pin_memory()
            self.batch_text_b_len = self.batch_text_b_len.pin_memory()
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            return self

    def collate_wrapper(batch):
        text_a_list = []
        text_b_list = []
        text_a_len = []
        text_b_len = []
        text_a_b_list = []
        text_a_max_length = 0
        text_b_max_length = 0
        for batch_data in batch:
            text_a_b_list.append((batch_data[0], batch_data[1]))
            text_a_list.append(batch_data[0])
            text_b_list.append(batch_data[1])

            text_a_word_list = batch_data[0].split()
            text_b_word_list = batch_data[1].split()

            text_a_len.append(len(text_a_word_list))
            text_b_len.append(len(text_b_word_list))

            text_a_max_length = len(text_a_word_list) if len(text_a_word_list) > text_a_max_length else text_a_max_length
            text_b_max_length = len(text_b_word_list) if len(text_b_word_list) > text_b_max_length else text_b_max_length

        label_list = LongTensor([batch_data[2] for batch_data in batch])
        
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, label_list, **ret)

    # TODO: to support num_workers
    predict_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False)#, num_workers=4)

    return predict_loader



def prepare_bert_mlm_dataset(path: str, tokenizer, batch_size=128):
    org_data = prepare_qqsim_data(path)
    sentence_list = []
    for sentence in chain(org_data[0], org_data[1], org_data[2]):
        if type(sentence) is str and len(sentence) > 1:
            sentence_list.append(sentence)

    data_set = OrgDataSet(sentence_list)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    def collate_wrapper(batch):
        text_a_list = [batch_data[0] for batch_data in batch]
        data = tokenizer.batch_encode_plus(text_a_list, padding=True, return_tensors="pt")
        return data_collator(tokenizer, data)

    return DataLoader(data_set, batch_size=batch_size, pin_memory=True, collate_fn=collate_wrapper)  # , num_workers=4)


def create_datafile_for_mlm(path_list: list, output_path: str):
    sentence_list = []
    for path in path_list:
        org_data = prepare_qqsim_data(path)
        for sentence in chain(org_data[0], org_data[1], org_data[2]):
            if type(sentence) is str and len(sentence) > 1:
                sentence_list.append(sentence)

    with open(output_path, mode='w') as f:
        f.write('\r\n'.join(sentence_list))
    pass


def create_double_sentence_datafile_for_mlm(path_list: list, output_path: str):
    sentence_list = []
    for path in path_list:
        org_data = prepare_qqsim_data(path)
        for sentence_a, sentence_b in zip(org_data[0], org_data[1]):
            if len(sentence_a) > 1 and len(sentence_b) > 1:
                sentence_list.append('\t'.join([sentence_a, sentence_b]))

    with open(output_path, mode='w') as f:
        f.write('\r\n'.join(sentence_list))
    pass


class DoubleSentenceTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    data = line.split('\t')
                    lines.append((data[0], data[1]))
                    lines.append((data[1], data[0]))

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class RandomDoubleSentenceTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    data = line.split('\t')
                    self.lines.append(data[0])
                    self.lines.append(data[1])

        # batch_encoding = tokenizer(self.lines, add_special_tokens=True, truncation=True, max_length=block_size)
        # self.examples = batch_encoding["input_ids"]
        # self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return int(len(self.lines) / 2)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        random_line = [(self.lines[random.randint(0, len(self.lines)-1)], self.lines[random.randint(0, len(self.lines)-1)])]
        random_line_encoding = self.tokenizer(random_line, add_special_tokens=True, truncation=True, max_length=self.block_size)
        ret = random_line_encoding["input_ids"][0]
        ret = {"input_ids": torch.tensor(ret, dtype=torch.long)}
        return ret


