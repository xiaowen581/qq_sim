import os
from model.model_bert import MyBertForSequenceClassification10 as MyBertForSequenceClassification
from transformers import AdamW #, RobertaForSequenceClassification as MyBertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from nlp_utils.dataset import SentenceLabelDataLoader
from torch import LongTensor, zeros, argmax, device, save, load, no_grad
from torch.cuda import is_available as cuda_is_available
from pycm import ConfusionMatrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_linear_schedule_with_warmup
import argparse
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from nlp_utils.tokenizer import WhiteSpaceTokenizer
from nlp_utils.dataset import prepare_qqsim_train_test_org_data_loader, prepare_qqsim_predict_org_data_loader, prepare_qqsim_train_test_tensor_data_loader
from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn.functional import softmax

device = device("cuda:0" if cuda_is_available() else "cpu")
#device = device("cpu")


class Config:
    last_model_path = ''
    pretrained_model_folder = '../user_data/pretrained/self_nezha_bert_double_sentence'
    train_batch_size = 128
    test_batch_size = 128
    predict_batch_size = 128
    epoch = 5
    test_model_num = 10
    log_dir = './output'
    model_folder = './output'
    num_labels = 2


config = Config()

def train(config, train_data_loader, test_data_loader):
    writer = SummaryWriter(log_dir=config.log_dir)

    model = MyBertForSequenceClassification.from_pretrained(config.pretrained_model_folder, config.num_labels)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=config.epoch*len(train_data_loader)
    )

    step = 0
    right_str = '0/0'
    test_loss_map = {}
    with tqdm(total=config.epoch, desc='epoch') as epoch_bar:
        for epoch_num in range(config.epoch):
            model.train()
            epoch_bar.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'])
            # print("start epoch: {}".format(epoch_num), flush=True)
            total_batch_num = len(train_data_loader)
            with tqdm(total=total_batch_num, desc='batch') as batch_bar:
                for batch_num, (batch_data) in enumerate(train_data_loader):
                    step += 1

                    batch_token_text_a = batch_data.batch_token_text_a.to(device)
                    batch_token_text_b = batch_data.batch_token_text_b.to(device)
                    batch_text_a_len = batch_data.batch_text_a_len.to(device)
                    batch_text_b_len = batch_data.batch_text_b_len.to(device)

                    batch_ids = batch_data.batch_input_ids.to(device)
                    batch_token_type = None if batch_data.batch_token_type_ids is None else batch_data.batch_token_type_ids.to(device)
                    batch_mask = batch_data.batch_attention_mask.to(device)
                    batch_label = batch_data.batch_label_list.to(device)
                    optimizer.zero_grad()
                    ret = model(
                        input_ids=batch_ids,
                        token_type_ids=batch_token_type,
                        attention_mask=batch_mask,
                        labels=batch_label,
                    )

                    loss = ret.loss
                    logits = ret.logits

                    loss.backward()

                    clip_grad_norm_(model.parameters(), 1.0)


                    optimizer.step()
                    scheduler.step()
                    writer.add_scalar("Train/loss", loss.item(), step)

                    if batch_num % 10 == 0:
                        predict = argmax(logits, 1)
                        right = batch_label
                        matrix = ConfusionMatrix(right.cpu().detach().numpy(), predict.cpu().detach().numpy())
                        acc = matrix.overall_stat['Overall ACC'] # ACC Macro

                        p = predict.detach().cpu().numpy().tolist()
                        r = right.detach().cpu().numpy().tolist()

                        all = len(batch_data.batch_input_ids)
                        right = 0
                        for i in range(all):
                            if p[i] == r[i]:
                                right += 1
                        right_str = '{}/{}'.format(right, all)

                        writer.add_scalar("Train/Overall ACC ", acc, step)

                    batch_bar.update(1)
                    batch_bar.set_postfix(loss=loss.item(), hit=right_str)

            model_name = os.path.join(config.model_folder, "pytorch_model.bin")
            rename_name_format = os.path.join(config.model_folder, "pytorch_model_{}_{}.bin")

            if os.path.exists(model_name):
                rename_name = rename_name_format.format(epoch_num, test_loss_map[epoch_num] if epoch_num > 0 else 'before')
                os.rename(model_name, rename_name)

            model.save_pretrained(config.model_folder)

            epoch_bar.update(1)

            loss = eval(model, test_data_loader, is_predict=False, writer=writer, step=step)
            test_loss_map[epoch_num + 1] = loss

    writer.close()
    print(test_loss_map)


def test(config, data_iter, is_predict=False):
    # test
    model = MyBertForSequenceClassification.from_pretrained(config.model_folder, config.num_labels)
    return eval(model, data_iter, is_predict)


def eval(model, data_iter, is_predict=False, writer=None, step=None):
    if len(data_iter) == 0:
        return

    model.to(device)
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_prob = np.array([], dtype=int)

    with no_grad():
        for batch_data in data_iter:
            labels_all = np.append(labels_all, batch_data.batch_label_list.detach().numpy())

            batch_token_text_a = batch_data.batch_token_text_a.to(device)
            batch_token_text_b = batch_data.batch_token_text_b.to(device)
            batch_ids = batch_data.batch_input_ids.to(device)
            batch_token_type = None if batch_data.batch_token_type_ids is None else batch_data.batch_token_type_ids.to(device)
            batch_mask = batch_data.batch_attention_mask.to(device)
            batch_label = None if is_predict is True else batch_data.batch_label_list.to(device)

            ret = model(input_ids=batch_ids, token_type_ids=batch_token_type, attention_mask=batch_mask, labels=batch_label)

            loss = ret.loss
            logits = ret.logits
            if not is_predict:
                loss_total += loss.item()

            predict_prob = np.append(predict_prob, [logit[1] for logit in softmax(logits, dim=1).cpu().numpy()])
            predict = argmax(logits, 1).cpu().numpy()
            predict_all = np.append(predict_all, predict)

    if is_predict:
        np.savetxt('result.tsv', predict_prob, fmt='%.03f', delimiter='\r\n')
        os.system('zip result.zip {}'.format('result.tsv'))
        return None
    else:
        matrix = ConfusionMatrix(labels_all.flatten(), predict_all.flatten())
        print('Test loss: {}'.format(loss_total / len(data_iter)))
        print(matrix)
        if writer:
            writer.add_scalar("Test/Overall ACC", matrix.overall_stat['Overall ACC'], step)
            writer.add_scalar("Test/loss", (loss_total / len(data_iter)), step)
        return round(loss_total / len(data_iter), 4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true", default=False, help='train')
    parser.add_argument('--test', action="store_true", default=False, help='test')
    parser.add_argument('--predict', action="store_true", default=False, help='predict')
    args = parser.parse_args()

    config = Config()

    never_split_token = [str(num) for num in range(30000)]

    if args.train:
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_folder, do_basic_tokenize=True)
        tokenizer.save_pretrained(config.model_folder)
        train_data_loader, test_data_loader = prepare_qqsim_train_test_org_data_loader(
            './data/qq_number/gaiic_track3_round1_train_20210228.tsv',
            tokenizer=tokenizer,
            batch_size=config.train_batch_size,
            test_scale=0.1,
        )

        train(config, train_data_loader, test_data_loader)

    elif args.test:
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_folder, do_basic_tokenize=True)
        train_data_loader, test_data_loader = prepare_qqsim_train_test_org_data_loader(
            './data/qq_number/gaiic_track3_round1_train_20210228.tsv',
            tokenizer=tokenizer,
            batch_size=config.test_batch_size,
            test_scale=1.0,
        )
        config.pretrained_model_folder = config.model_folder
        test(config, test_data_loader)
    elif args.predict:
        tokenizer = BertTokenizer.from_pretrained(config.model_folder, do_basic_tokenize=True)
        predict_data_loader = prepare_qqsim_predict_org_data_loader(
            './data/qq_number/gaiic_track3_round1_testA_20210228.tsv',
            tokenizer=tokenizer,
            batch_size=config.predict_batch_size,
        )
        test(config, predict_data_loader, is_predict=True)

    pass
