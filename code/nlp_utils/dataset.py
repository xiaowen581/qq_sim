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
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from collections import defaultdict


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
        # if len(value) > 2:
        same_sentence = [get_sentence_by_id(data_label_1, index) for index in value]
        same_sentences_list.append(same_sentence)

    return same_sentences_list


def create_enhance_data(data):
    # TODO： 隐含label1必须为稀有标签，不然会导致死循环
    org_a_0 = []
    org_b_0 = []
    org_label_0 = []

    for (sent_a, sent_b, label) in zip(*data):
        if label == 0:
            org_a_0.append(sent_a)
            org_b_0.append(sent_b)
            org_label_0.append(0)

    org_label_0_len = len(org_a_0)
    org_label_1_len = len(data[0]) - org_label_0_len

    label_rate = org_label_0_len / org_label_1_len

    enhance_a_1 = []
    enhance_b_1 = []
    enhance_label_1 = []

    enhance_a_0 = []
    enhance_b_0 = []
    enhance_label_0 = []
    same_sentences_list = get_same_meaning_sentence(data)
    for same_sentences in same_sentences_list:
        for i in range(len(same_sentences)):
            for j in range(i+1, len(same_sentences)):
                enhance_a_1.append(same_sentences[i])
                enhance_b_1.append(same_sentences[j])
                enhance_label_1.append(int(1))

            if len(same_sentences) > 2:
                enhance_a_0.append(same_sentences[i])
                sentence_label_0 = data[random.randint(0, 1)][random.randint(0, len(data[0])-1)]
                while sentence_label_0 in same_sentences_list:
                    sentence_label_0 = data[random.randint(0, 1)][random.randint(0, len(data[0]) - 1)]
                enhance_b_0.append(sentence_label_0)
                enhance_label_0.append(int(0))

    add_0_len = label_rate * len(enhance_label_1) - org_label_0_len - len(enhance_label_0)
    for _ in range(int(add_0_len)):
        enhance_a_0.append(data[random.randint(0, 1)][random.randint(0, len(data[0])-1)])
        enhance_b_0.append(data[random.randint(0, 1)][random.randint(0, len(data[0])-1)])
        enhance_label_0.append(int(0))

    ret_a = org_a_0 + enhance_a_1 + enhance_a_0
    ret_b = org_b_0 + enhance_b_1 + enhance_b_0
    ret_label = org_label_0 + enhance_label_1 + enhance_label_0
    return ret_a, ret_b, ret_label


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


def prepare_bert_data(path: str, tokenizer, padding_max_length=None):
    data = read_file_as_str(path)

    # data = data[:64]

    input_ids_ret = []
    token_type_ids_ret = []
    attention_masks_ret = []
    labels_ret = []
    for index, line in enumerate(data):
        sentence, label = line.strip().split('\t')
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        if padding_max_length is None:
            sentence_feature = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
        else:
            sentence_feature = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt",
                                                     truncation=True, max_length=padding_max_length,
                                                     pad_to_max_length=True)
        input_ids_ret.append(sentence_feature['input_ids'])
        token_type_ids_ret.append(sentence_feature['token_type_ids'])
        attention_masks_ret.append(sentence_feature['attention_mask'])
        labels_ret.append(int(label))

    return input_ids_ret, token_type_ids_ret, attention_masks_ret, labels_ret


def prepare_single_sentence_bert_tensor(sentence: str, tokenizer, padding_max_length=None):
    if padding_max_length is None:
        sentence_feature = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
    else:
        sentence_feature = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt",
                                                 truncation=True, max_length=padding_max_length,
                                                 pad_to_max_length=True)
    return sentence_feature['input_ids'], sentence_feature['token_type_ids'], sentence_feature['attention_mask']


def prepare_sentence_pair_bert_tensor(sentence_a: str, sentence_b: str, tokenizer, padding_max_length=None):
    if padding_max_length is None:
        sentence_feature = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True, return_tensors="pt")
    else:
        sentence_feature = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True, return_tensors="pt",
                                                 truncation=True, max_length=padding_max_length,
                                                 pad_to_max_length=True)
    return sentence_feature['input_ids'], sentence_feature['token_type_ids'], sentence_feature['attention_mask']


class SentenceLabelDataLoader:
    def __init__(self, path, config, batch_size=64, padding=256, shuffle=True):
        # TODO: do prepare_bert_data in DataLoader worker
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_folder)
        input_ids, token_type_ids, attention_masks, labels = prepare_bert_data(path, tokenizer, padding)
        input_ids = cat(input_ids, dim=0)
        token_type_ids = cat(token_type_ids, dim=0)
        attention_masks = cat(attention_masks, dim=0)
        labels = LongTensor(labels)

        train_ids = TensorDataset(input_ids, token_type_ids, attention_masks, labels)

        self.dataLoader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    def next_batch(self):
        for step, (batch_ids, batch_token_type, batch_mask, batch_label) in enumerate(self.dataLoader):
            yield batch_ids, batch_token_type, batch_mask, batch_label

    def __len__(self):
        return len(self.dataLoader)

    def get_sample(self):
        ret = iter(self.dataLoader).__next__()
        return ret
        #return self.dataLoader.next()
        # return self.dataLoader.dataset[0][0], self.dataLoader.dataset[1][0], self.dataLoader.dataset[2][0], self.dataLoader.dataset[3][0]


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


class TokenizedDataSet(Dataset):

    def __init__(self, data_dict: dict):
        super(TokenizedDataSet, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


def prepare_qqsim_data(path: str, is_train=False, is_sort_by_length=False):
    data = read_file_as_str(path)
    sentence_a = []
    sentence_b = []
    labels = []

    if is_sort_by_length:
        data = sorted(data, key=lambda i: len(i))

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


def prepare_qqsim_train_test_org_data_loader(path: str, tokenizer, batch_size=64, test_scale=0.2, need_dualization=True):
    org_a, org_b, org_labels = prepare_qqsim_data(path, is_train=True)

    test_len = int(len(org_a) * test_scale)
    train_len = len(org_a) - test_len

    train_org_a, train_org_b, train_org_labels = org_a[:train_len], org_b[:train_len], org_labels[:train_len]
    test_a, test_b, test_labels = org_a[train_len:], org_b[train_len:], org_labels[train_len:]

    # 数据增强
    enhance_a, enhance_b, enhance_label = create_enhance_data((train_org_a, train_org_b, train_org_labels))
    train_org_a = enhance_a
    train_org_b = enhance_b
    train_org_labels = enhance_label

    if need_dualization:
        data_a = train_org_a + train_org_b
        data_b = train_org_b + train_org_a
        data_labels = train_org_labels + train_org_labels
    else:
        data_a = train_org_a
        data_b = train_org_b
        data_labels = train_org_labels

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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
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


def prepare_qqsim_train_test_org_data_loader_no_data_enhance(path: str, tokenizer, batch_size=64, test_scale=0.2, is_sort_by_length=False):
    org_a, org_b, org_labels = prepare_qqsim_data(path, is_train=True, is_sort_by_length=is_sort_by_length)

    test_len = int(len(org_a) * test_scale)
    train_len = len(org_a) - test_len

    # train_org_a, train_org_b, train_org_labels = org_a[:train_len], org_b[:train_len], org_labels[:train_len]
    # test_a, test_b, test_labels = org_a[train_len:], org_b[train_len:], org_labels[train_len:]

    train_org_a = []
    train_org_b = []
    train_org_labels = []
    test_a = []
    test_b = []
    test_labels = []
    divider = int(1/test_scale)
    for index, (a, b, label) in enumerate(zip(org_a, org_b, org_labels)):
        if index % divider is 0:
            test_a.append(a)
            test_b.append(b)
            test_labels.append(label)
        else:
            train_org_a.append(a)
            train_org_b.append(b)
            train_org_labels.append(label)

    train_data_set = OrgDataSet(train_org_a, train_org_b, train_org_labels)
    test_data_set = OrgDataSet(test_a, test_b, test_labels)

    num_0 = num_1 = 0
    for label in train_org_labels:
        if label is 0:
            num_0 += 1
        else:
            num_1 += 1

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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, label_list, **ret)

    # TODO: to support num_workers
    train_loader = DataLoader(train_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=False if is_sort_by_length else True)#, num_workers=4)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=False)#, num_workers=4)

    return train_loader, test_loader


def prepare_qqsim_train_test_org_data_loader_no_data_enhance_mgp(path: str, tokenizer, mgp, batch_size=64, test_scale=0.2):
    org_a, org_b, org_labels = prepare_qqsim_data(path, is_train=True)

    test_len = int(len(org_a) * test_scale)
    train_len = len(org_a) - test_len

    train_org_a, train_org_b, train_org_labels = org_a[:train_len], org_b[:train_len], org_labels[:train_len]
    test_a, test_b, test_labels = org_a[train_len:], org_b[train_len:], org_labels[train_len:]

    train_mgp_text_a = mgp.batch_score(train_org_a)
    train_mgp_text_b = mgp.batch_score(train_org_b)
    test_mgp_text_a = mgp.batch_score(test_a)
    test_mgp_text_b = mgp.batch_score(test_b)

    train_data_set = OrgDataSet(train_org_a, train_org_b, train_org_labels, train_mgp_text_a, train_mgp_text_b)
    test_data_set = OrgDataSet(test_a, test_b, test_labels, test_mgp_text_a, test_mgp_text_b)

    num_0 = num_1 = 0
    for label in train_org_labels:
        if label is 0:
            num_0 += 1
        else:
            num_1 += 1

    print('num_0:{}, num_1:{}'.format(num_0, num_1))

    class CollateBatch:
        def __init__(self, token_text_a, token_text_b, text_a_len, text_b_len,  mgp_text_a, mgp_text_b, label_list, input_ids, token_type_ids, attention_mask):
            self.batch_token_text_a = token_text_a
            self.batch_token_text_b = token_text_b
            self.batch_text_a_len = text_a_len
            self.batch_text_b_len = text_b_len
            self.batch_input_ids = input_ids
            self.batch_token_type_ids = token_type_ids
            self.batch_attention_mask = attention_mask
            self.batch_label_list = label_list
            self.batch_mgp_text_a = mgp_text_a
            self.batch_mgp_text_b = mgp_text_b

        def pin_memory(self):
            self.batch_token_text_a = self.batch_token_text_a.pin_memory()
            self.batch_token_text_b = self.batch_token_text_b.pin_memory()
            self.batch_text_a_len = self.batch_text_a_len.pin_memory()
            self.batch_text_b_len = self.batch_text_b_len.pin_memory()
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            self.batch_mgp_text_a = self.batch_mgp_text_a.pin_memory()
            self.batch_mgp_text_b = self.batch_mgp_text_b.pin_memory()
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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        batch_mgp_text_a = torch.FloatTensor([batch_data[3] for batch_data in batch])
        batch_mgp_text_b = torch.FloatTensor([batch_data[4] for batch_data in batch])

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, batch_mgp_text_a, batch_mgp_text_b, label_list, **ret)

    # TODO: to support num_workers
    train_loader = DataLoader(train_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=True)#, num_workers=4)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=False)#, num_workers=4)

    return train_loader, test_loader

def prepare_qqsim_train_test_org_data_loader_mixed_test(path: str, tokenizer, feature_for_lstm=False, batch_size=64, test_scale=0.2):
    org_a, org_b, org_labels = prepare_qqsim_data(path, is_train=True)

    # 数据增强
    enhance_a, enhance_b, enhance_label = create_enhance_data((org_a, org_b, org_labels))
    org_a += enhance_a
    org_b += enhance_b
    org_labels += enhance_label

    data_a = org_a + org_b
    data_b = org_b + org_a
    data_labels = org_labels + org_labels

    data_set = OrgDataSet(data_a, data_b, data_labels)
    test_length = int(len(data_set) * test_scale)
    length = [len(data_set) - test_length, test_length]
    train_indices, test_indices = random_split(data_set, length)

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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")

        if feature_for_lstm:
            batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
            batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
            batch_text_a_len = torch.LongTensor(text_a_len)
            batch_text_b_len = torch.LongTensor(text_b_len)
        else:
            batch_token_text_a = None
            batch_token_text_b = None
            batch_text_a_len = None
            batch_text_b_len = None

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, label_list, **ret)

    # TODO: to support num_workers
    train_loader = DataLoader(train_indices, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True)#, num_workers=4)
    test_loader = DataLoader(test_indices, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True)#, num_workers=4)

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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, label_list, **ret)

    # TODO: to support num_workers
    predict_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False)#, num_workers=4)

    return predict_loader


def prepare_qqsim_predict_org_data_loader_mgp(path: str, tokenizer, mgp, batch_size=1):
    org_a, org_b, org_label = prepare_qqsim_data(path)

    mgp_text_a = mgp.batch_score(org_a)
    mgp_text_b = mgp.batch_score(org_b)

    data_set = OrgDataSet(org_a, org_b, org_label, mgp_text_a, mgp_text_b)

    class CollateBatch:
        def __init__(self, token_text_a, token_text_b, text_a_len, text_b_len,  mgp_text_a, mgp_text_b, label_list, input_ids, token_type_ids, attention_mask):
            self.batch_token_text_a = token_text_a
            self.batch_token_text_b = token_text_b
            self.batch_text_a_len = text_a_len
            self.batch_text_b_len = text_b_len
            self.batch_input_ids = input_ids
            self.batch_token_type_ids = token_type_ids
            self.batch_attention_mask = attention_mask
            self.batch_label_list = label_list
            self.batch_mgp_text_a = mgp_text_a
            self.batch_mgp_text_b = mgp_text_b

        def pin_memory(self):
            self.batch_token_text_a = self.batch_token_text_a.pin_memory()
            self.batch_token_text_b = self.batch_token_text_b.pin_memory()
            self.batch_text_a_len = self.batch_text_a_len.pin_memory()
            self.batch_text_b_len = self.batch_text_b_len.pin_memory()
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            self.batch_mgp_text_a = self.batch_mgp_text_a.pin_memory()
            self.batch_mgp_text_b = self.batch_mgp_text_b.pin_memory()
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
        # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
        ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
        batch_token_text_a = tokenizer.batch_encode_plus(text_a_list, add_special_tokens=False, max_length=text_a_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_token_text_b = tokenizer.batch_encode_plus(text_b_list, add_special_tokens=False, max_length=text_b_max_length, pad_to_max_length=True, return_tensors="pt")['input_ids']
        batch_text_a_len = torch.LongTensor(text_a_len)
        batch_text_b_len = torch.LongTensor(text_b_len)

        batch_mgp_text_a = torch.FloatTensor([batch_data[3] for batch_data in batch])
        batch_mgp_text_b = torch.FloatTensor([batch_data[4] for batch_data in batch])

        return CollateBatch(batch_token_text_a, batch_token_text_b, batch_text_a_len, batch_text_b_len, batch_mgp_text_a, batch_mgp_text_b, label_list, **ret)

    # TODO: to support num_workers
    predict_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False)#, num_workers=4)

    return predict_loader


def prepare_qqsim_train_test_tensor_data_loader(path: str, tokenizer, batch_size=64, test_scale=0.2, max_length=128):
    org_data = prepare_qqsim_data(path)
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    for index in range(len(org_data[0])):
        sentence_a = org_data[0][index]
        sentence_b = org_data[1][index]
        label = org_data[2][index]

        input_ids, attention_mask = get_token(tokenizer, sentence_a, sentence_b, int(max_length / 2))
        input_ids_list.append(LongTensor(input_ids))
        attention_mask_list.append(LongTensor(attention_mask))
        label_list.append(label)

    input_ids_list = cat(input_ids_list, dim=0).reshape(len(input_ids_list), -1)
    attention_mask_list = cat(attention_mask_list, dim=0).reshape(len(input_ids_list), -1)
    label_list = LongTensor(label_list)

    data_set = TensorDataset(input_ids_list, attention_mask_list, label_list)

    test_length = int(len(data_set) * test_scale)
    length = [len(data_set) - test_length, test_length]
    train_indices, test_indices = random_split(data_set, length)

    class CollateBatch:
        def __init__(self, data):
            self.batch_input_ids = cat([batch_data[0] for batch_data in data], dim=0).reshape([len(data), -1])
            # self.batch_token_type_ids = cat([cat([zeros(64, dtype=long), ones(64, dtype=long)], dim=0) for _ in range(len(data))], dim=0).reshape([len(data), -1])
            self.batch_token_type_ids = None
            self.batch_attention_mask = cat([batch_data[1] for batch_data in data], dim=0).reshape([len(data), -1])
            self.batch_label_list = LongTensor([batch_data[2] for batch_data in data])
            pass

        def pin_memory(self):
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            # self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            return self

    def collate_wrapper(batch):
        return CollateBatch(batch)

    # TODO: to support num_workers
    train_loader = DataLoader(train_indices, batch_size=batch_size, pin_memory=True, collate_fn=collate_wrapper)#, num_workers=4)
    test_loader = DataLoader(test_indices, batch_size=batch_size, pin_memory=True, collate_fn=collate_wrapper)#, num_workers=4)

    return train_loader, test_loader


def prepare_qqsim_predict_tensor_data_loader(path: str, tokenizer, batch_size=1):
    org_data = prepare_qqsim_data(path)
    data_set = OrgDataSet(*org_data)

    class CollateBatch:
        def __init__(self, data):
            text_a_b_list = [(batch_data[0], batch_data[1]) for batch_data in data]
            label_list = LongTensor([batch_data[2] for batch_data in data])
            # TODO: set max_length to ??? to get rid of cuda out of memory when training on 1080ti(11gb)
            ret = tokenizer.batch_encode_plus(text_a_b_list, padding=True, return_tensors="pt")
            self.batch_input_ids = ret.input_ids
            self.batch_token_type_ids = ret.token_type_ids
            self.batch_attention_mask = ret.attention_mask
            self.batch_label_list = label_list
            pass

        def pin_memory(self):
            self.batch_input_ids = self.batch_input_ids.pin_memory()
            self.batch_token_type_ids = self.batch_token_type_ids.pin_memory()
            self.batch_attention_mask = self.batch_attention_mask.pin_memory()
            self.batch_label_list = self.batch_label_list.pin_memory()
            return self

    def collate_wrapper(batch):
        return CollateBatch(batch)

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


def prepare_double_sentence_tokenized_mlm_dataset(mlm_file_path, tokenizer: BertTokenizer) -> dict:
    org_data = prepare_qqsim_data(mlm_file_path, header=None, sep='\t')

    inputs = defaultdict(list)
    for i, row in org_data:
        sentence_a, sentence_b = row[0], row[1]
        inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class DoubleSentenceTokenizedTextDataset(Dataset):
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

        self.examples = defaultdict(list)
        for data in lines:
            batch_encoding = tokenizer.encode_plus(data[0], data[1], add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True)
            self.examples['input_ids'].append(batch_encoding['input_ids'])
            self.examples['token_type_ids'].append(batch_encoding['token_type_ids'])
            self.examples['attention_mask'].append(batch_encoding['attention_mask'])

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, i):
        return (self.examples['input_ids'][i], self.examples['token_type_ids'][i], self.examples['attention_mask'][i])


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


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForNgramLanguageModeling:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)
            token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _whole_word_mask(self, input_ids_list: List[str], max_seq_len: int, max_predictions=512):
        cand_indexes = []
        for (i, token) in enumerate(input_ids_list):
            if (token == str(self.tokenizer.cls_token_id)
                    or token == str(self.tokenizer.sep_token_id)):
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_ids_list) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids_list))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def whole_word_mask(self, input_ids_list: List[list], max_seq_len: int) -> torch.Tensor:
        mask_labels = []
        for input_ids in input_ids_list:
            wwm_id = random.choices(range(len(input_ids)), k=int(len(input_ids) * 0.2))
            input_id_str = [f'##{id_}' if i in wwm_id else str(id_) for i, id_ in enumerate(input_ids)]
            mask_label = self._whole_word_mask(input_id_str, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        # generate wwm segments
        batch_mask = self.whole_word_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


if __name__ == '__main__':
    # import torch
    # # Create dummy data with class imbalance 99 to 1
    # numDataPoints = 1000
    # data_dim = 5
    # bs = 100
    # data = torch.randn(numDataPoints, data_dim)
    # target = torch.cat((torch.zeros(int(numDataPoints * 0.99), dtype=torch.long),
    #                     torch.ones(int(numDataPoints * 0.01), dtype=torch.long)))
    #
    # print('target train 0/1: {}/{}'.format(
    #     (target == 0).sum(), (target == 1).sum()))
    #
    # # Create subset indices
    # subset_idx = torch.cat((torch.arange(100), torch.arange(-5, 0)))
    #
    # # Compute samples weight (each sample should get its own weight)
    # class_sample_count = torch.tensor(
    #     [(target[subset_idx] == t).sum() for t in torch.unique(target, sorted=True)])
    # weight = 1. / class_sample_count.float()
    # samples_weight = torch.tensor([weight[t] for t in target[subset_idx]])
    #
    # # Create sampler, dataset, loader
    #
    # train_dataset = torch.utils.data.TensorDataset(
    #     data[subset_idx], target[subset_idx])
    #
    # test_length = int(len(train_dataset) * 0.2)
    # length = [len(train_dataset) - test_length, test_length]
    # train_indices, test_indices = random_split(train_dataset, length)
    #
    # train_loader = DataLoader(
    #     train_indices, batch_size=bs, num_workers=1)
    #
    # # Iterate DataLoader and check class balance for each batch
    # for i, (x, y) in enumerate(train_loader):
    #     print("batch index {}, 0/1: {}/{}".format(i, (y == 0).sum(), (y == 1).sum()))

    class Config:
        train_batch_size = 64
        test_batch_size = 64
        pretrained_model_folder = './pretrained_model/hfl/chinese-bert-wwm-ext'

    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_folder)

    # #train = SentenceLabelDataLoader('./data/bi_classify/train.txt')
    # test = SentenceLabelDataLoader('./data/bi_classify/test.txt', config)
    # # dev = SentenceLabelDataLoader('./data/bi_classify/dev.txt', batch_size=4)
    # #
    # # for a, b, c, d in dev.next_batch():
    # #     print('a: {}, b: {}, c: {}, d:{}'.format(a, b, c, d))
    #
    # for index, data in enumerate(test.dataLoader):
    #     print(index)

    # org_data = prepare_qqsim_data('./data/qq/gaiic_track3_round1_train_20210220.tsv')
    # data_set = OrgDataSet(*org_data)
    #
    # test_length = int(len(data_set) * 0.2)
    # length = [len(data_set) - test_length, test_length]
    # train_indices, test_indices = random_split(data_set, length)
    # train_sampler = SubsetRandomSampler(train_indices)
    # train_loader = DataLoader(data_set, batch_size=64,  # collate_fn=collate_fn,
    #                           sampler=train_sampler)
    # for index, data in enumerate(train_loader):
    #     print(index)

    # train_loader, test_loader = prepare_qqsim_train_test_org_data_loader('./data/qq/gaiic_track3_round1_train_20210220.tsv', tokenizer)
    # train_org_data_set, test_org_data_set = prepare_qqsim_train_test_org_data_set('./data/qq/gaiic_track3_round1_train_20210220.tsv')
    # train_data_set = SentencePairLabelDataLoader(train_org_data_set, config)
    # test_data_set = SentenceLabelDataLoader(test_org_data_set, config)
    # for index, data in enumerate(train_loader):
    #     print(data)
    #     break

    train_loader, test_loader = prepare_qqsim_train_test_tensor_data_loader('./data/qq/test.tsv', tokenizer)
    pass

