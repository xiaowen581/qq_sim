import os
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, AdamW
import torch
from nlp_utils.dataset import DoubleSentenceTokenizedTextDataset, create_double_sentence_datafile_for_mlm, DataCollatorForNgramLanguageModeling
from transformers import Trainer, TrainingArguments
from collections import defaultdict
from nezha.modeling_nezha import NeZhaForMaskedLM as BertForMaskedLM
from nezha.configuration_nezha import NeZhaConfig as BertConfig

pretrained_model_nezha_base = '../user_data/pretrained/nezha-cn-base'
pretrained_model_folder = '../user_data/pretrained/self_nezha_bert_double_ngram_sentence_green'

train_file_path = './data/qq_number/gaiic_track3_round1_train_20210228.tsv'
test_file_path = './data/qq_number/gaiic_track3_round1_testA_20210228.tsv'
mlm_file_path = './data/qq_number/mlm.txt'
mlm_double_file_path = './data/qq_number/mlm_double_sentence.txt'


def get_dict(data):
    words_dict = defaultdict(int)
    for i in tqdm(range(data.shape[0])):
        text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
        for c in text:
            words_dict[c] += 1
    return words_dict


def train():
    ##训练集和测试集造字典
    train = pd.read_csv('./data/qq_number/gaiic_track3_round1_train_20210228.tsv', sep='\t',
                        names=['text_a', 'text_b', 'label'])
    test = pd.read_csv('./data/qq_number/gaiic_track3_round1_testA_20210228.tsv', sep='\t',
                       names=['text_a', 'text_b', 'label'])
    test['label'] = 0
    test_dict = get_dict(test)
    train_dict = get_dict(train)
    word_dict = list(test_dict.keys()) + list(train_dict.keys())
    word_dict = set(word_dict)
    word_dict = set(map(int, word_dict))
    word_dict = list(word_dict)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    WORDS = special_tokens + word_dict

    if not os.path.exists(pretrained_model_folder):
        os.mkdir(pretrained_model_folder)

    pd.Series(WORDS).to_csv(os.path.join(pretrained_model_folder, 'vocab.txt'), header=False, index=0)

    if not os.path.exists(mlm_double_file_path):
        create_double_sentence_datafile_for_mlm([test_file_path, train_file_path], mlm_double_file_path)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = BertForMaskedLM.from_pretrained(pretrained_model_nezha_base)

    model = model.to(device)

    dataset = DoubleSentenceTokenizedTextDataset(
        tokenizer=tokenizer,
        file_path=mlm_double_file_path,
        block_size=128,
    )

    data_collator = DataCollatorForNgramLanguageModeling(
        max_seq_len=32, tokenizer=tokenizer, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=pretrained_model_folder,
        overwrite_output_dir=True,
        num_train_epochs=60,
        per_device_train_batch_size=128,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        dataloader_num_workers=4,
        learning_rate=1e-4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(pretrained_model_folder)


def predict():
    from transformers import pipeline

    # fill-maskパイプラインの作成
    fill_mask = pipeline(
        "fill-mask",
        model=BertForMaskedLM.from_pretrained(pretrained_model_folder),
        tokenizer=pretrained_model_folder
    )
    print(fill_mask("1 2 [MASK] 4 5 6 7")) # 3
    print(fill_mask("12 23 [MASK] 123 59")) # 122


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true", default=False, help='train')
    parser.add_argument('--predict', action="store_true", default=False, help='predict')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict()

    pass

