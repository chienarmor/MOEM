import torch
from collections import defaultdict
from torch.utils.data import Dataset
from utils.const import *
import random
from itertools import permutations
from itertools import chain
from torch.utils.data import Sampler
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pdb

def parse_aste_tuple(_tuple, sent):
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # parse at
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # parse ot
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res


def read_line_examples_from_file(data_path,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if "unified" in task_name:
                _task, _data, line = line.split("\t")
                tasks.append(_task)
                datas.append(_data)
            else:
                tasks.append(task_name)
                datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels


def get_transformed_io_unified(data_path, task_name, data_name, data_type,
                               top_k, args):
    """
    The main function to transform input & target according to the task
    """
    """
    tasks：任务名称
    datas：数据集名称
    sents：句子内容
    labels：标记
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, task_name, data_name, lowercase=args.lowercase)
    sents = [s.copy() for s in sents]
    new_inputs, targets = [], []
    for task, data, sent, label in zip(tasks, datas, sents, labels):
        if data_type == "train" or (data_type == "test" and args.multi_path):
            new_input, target = get_new_sents([sent], label, data,
                                              top_k, task, args)
        # else:
        #     new_input, target = get_para_targets_dev([sent], [label], data,
        #                                              task, args)
        new_inputs.extend(new_input)
        targets.extend(target)

    # print("Ori sent size:", len(sents))
    # print("Input size:", len(new_inputs), len(targets))
    # print("Examples:")
    # print(new_inputs[:10])
    # print(targets[:10])

    return new_inputs, targets


def get_orders(task, data, args, sents, labels):
    ## uncomment to calculate orders from scratch
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device("cpu")
    # tokenizer = T5Tokenizer.from_pretrained("t5-base").to(device)
    # model = MyT5ForConditionalGenerationScore.from_pretrained(
    #     "t5-base").to(device)
    # optim_orders_all = choose_best_order_global(sents, labels, model,
    #                                         tokenizer, device,
    #                                         args.task)

    if args.single_view_type == 'rank':
        orders = optim_orders_all[task][data]
    elif args.single_view_type == 'rand':
        orders = [random.Random(args.seed).choice(
            optim_orders_all[task][data])]
    elif args.single_view_type == "heuristic":
        orders = heuristic_orders[task]
    return orders


def get_task_tuple(_tuple, task):
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp:
        sp = sentword2opinion[sp.lower()] if sp in sentword2opinion \
            else senttag2opinion[sp.lower()]  # 'POS' -> 'good'
    if at and at.lower() == 'null':  # for implicit aspect term
        at = 'it'

    return at, ac, sp, ot


def add_prompt(sent, orders, task, data_name, args):
    if args.multi_task:
        # add task and data prefix
        sent = [task, ":", data_name, ":"] + sent

    # add ctrl_token
    if args.ctrl_token == "none":
        pass
    elif args.ctrl_token == "post":
        sent = sent + orders
    elif args.ctrl_token == "pre":
        sent = orders + sent
    else:
        raise NotImplementedError
    return sent


def get_para_targets(sents, labels, data_name, data_type, top_k, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []
    if task in ['aste', 'tasd']:
        # at most 5 orders for triple tasks
        top_k = min(5, top_k)

    optim_orders = get_orders(task, data_name, args, sents, labels)[:top_k]

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]
        cur_sent_str = " ".join(cur_sent)

        # ASTE: parse at & ot
        if task == 'aste':
            assert len(label[0]) == 3
            parsed_label = []
            for _tuple in label:
                parsed_tuple = parse_aste_tuple(_tuple, sents[i])
                parsed_label.append(parsed_tuple)
            label = parsed_label

        # sort label by order of appearance
        # at, ac, sp, ot
        if args.sort_label and len(label) > 1:
            label_pos = {}
            for _tuple in label:
                at, ac, sp, ot = get_task_tuple(_tuple, task)

                # get last at / ot position
                at_pos = cur_sent_str.find(at) if at else -1
                ot_pos = cur_sent_str.find(ot) if ot else -1
                last_pos = max(at_pos, ot_pos)
                last_pos = 1e4 if last_pos < 0 else last_pos
                label_pos[tuple(_tuple)] = last_pos
            new_label = [
                list(k)
                for k, _ in sorted(label_pos.items(), key=lambda x: x[1])
            ]
            label = new_label

        quad_list = []
        for _tuple in label:
            at, ac, sp, ot = get_task_tuple(_tuple, task)
            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            token_end = 3

            element_list = []
            for key in optim_orders[0].split(" "):
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        for o in optim_orders:
            tar = []
            for each_q in quad_list:
                tar.append(each_q[o][1])

            targets.append(" [SSEP] ".join(tar))
            # add prompt
            new_sent = add_prompt(cur_sent, o.split(), task, data_name, args)
            new_sents.append(new_sent)

    return new_sents, targets


def get_new_sents(sents, labels, data_name, top_k, task, args):
    new_sents = []
    new_labels = []
    # data_name = 'rest15'
    optim_orders = get_orders(task, data_name, args, sents, labels)[:top_k]
    # optim_orders = [get_orders(task, data_name, args, sents, labels)[top_k]]
    for i in range(len(sents)):
        cur_sent = sents[i]
        for o in optim_orders:
            # add prompt
            new_sent = add_prompt(cur_sent, o.split(), task, data_name, args)
            new_sents.append(new_sent)
            new_labels.append(labels)
    return new_sents, new_labels


def ABSA_format(sentence, label, task):
    polarity2word = {'pos': "positive", "neg": "negative", "neu": "neutral"}
    # assert marker_order in ['aspect', 'opinion'] and len(label) != 0
    all_tri = []
    if task in ['asqp', 'acos']:
        for qua in label:
            if qua[0] == 'null':
                a = 'it'
            else:
                if len(qua[0]) == 1:
                    a = sentence[qua[0][0]]
                else:
                    st, ed = qua[0][0], qua[0][-1]
                    a = ' '.join(sentence[st: ed + 1])
            if qua[3] == 'null':
                b = 'null'
            else:
                if len(qua[3]) == 1:
                    b = sentence[qua[3][0]]
                else:
                    st, ed = qua[3][0], qua[3][-1]
                    b = ' '.join(sentence[st: ed + 1])
            s = qua[2]
            c = qua[1]
            all_tri.append((a, c, s, b))
    else:
        for tri in label:
            if len(tri[0]) == 1:
                a = sentence[tri[0][0]]
            else:
                st, ed = tri[0][0], tri[0][-1]
                a = ' '.join(sentence[st: ed + 1])
            if len(tri[1]) == 1:
                b = sentence[tri[1][0]]
            else:
                st, ed = tri[1][0], tri[1][-1]
                b = ' '.join(sentence[st: ed + 1])
            s = polarity2word[tri[2]]
            all_tri.append((a, b, s))
    label_strs = []
    for tuple in all_tri:
        permutation = []
        if task in ['asqp', 'acos']:
            permutation.append(sentence[-4])
            permutation.append(sentence[-3])
            permutation.append(sentence[-2])
            permutation.append(sentence[-1])
            mapping = {'[A]': tuple[0], '[C]': tuple[1], '[S]': tuple[2], '[O]': tuple[3]}
            label_list = [char + ' ' + mapping[char] for char in permutation]
            label_str = ' '.join(label_list)
            label_strs.append(label_str)
        else:
            permutation.append(sentence[-3])
            permutation.append(sentence[-2])
            permutation.append(sentence[-1])
            mapping = {'[A]': tuple[0], '[O]': tuple[1], '[S]': tuple[2]}
            label_list = [char + ' ' + mapping[char] for char in permutation]
            label_str = ' '.join(label_list)
            label_strs.append(label_str)
    # mapping = {'1': 'a', '2': 'b', '3': 'c'}
    # element1, element2, element3 = sentence[-3], sentence[-2], sentence[-1]
    # label_strs = []
    # for triplet in all_tri:
    #     label_str = ''
    #     if element1 == '[aspect]':
    #         label_str += '[aspect] ' + triplet[0]
    #         if element2 == '[opinion]':
    #             label_str += ' [opinion] ' + triplet[1]
    #             label_str += ' [sentiment] ' + triplet[2]
    #         else:
    #             label_str += ' [sentiment] ' + triplet[2]
    #             label_str += ' [opinion] ' + triplet[1]
    #     elif element1 == '[opinion]':
    #         label_str += '[opinion] ' + triplet[1]
    #         if element2 == '[aspect]':
    #             label_str += ' [aspect] ' + triplet[0]
    #             label_str += ' [sentiment] ' + triplet[2]
    #         else:
    #             label_str += ' [sentiment] ' + triplet[2]
    #             label_str += ' [aspect] ' + triplet[0]
    #     else:
    #         label_str += '[sentiment] ' + triplet[2]
    #         if element2 == '[aspect]':
    #             label_str += ' [aspect] ' + triplet[0]
    #             label_str += ' [opinion] ' + triplet[1]
    #         else:
    #             label_str += ' [opinion] ' + triplet[1]
    #             label_str += ' [aspect] ' + triplet[0]
    #     label_strs.append(label_str)
    # for triplet in all_tri:
    #     label_str = ''
    #     if element1 == '[A]':
    #         label_str += '[A] ' + triplet[0]
    #         if element2 == '[O]':
    #             label_str += ' [O] ' + triplet[1]
    #             label_str += ' [S] ' + triplet[2]
    #         else:
    #             label_str += ' [S] ' + triplet[2]
    #             label_str += ' [O] ' + triplet[1]
    #     elif element1 == '[O]':
    #         label_str += '[O] ' + triplet[1]
    #         if element2 == '[A]':
    #             label_str += ' [A] ' + triplet[0]
    #             label_str += ' [S] ' + triplet[2]
    #         else:
    #             label_str += ' [S] ' + triplet[2]
    #             label_str += ' [A] ' + triplet[0]
    #     else:
    #         label_str += '[S] ' + triplet[2]
    #         if element2 == '[A]':
    #             label_str += ' [A] ' + triplet[0]
    #             label_str += ' [O] ' + triplet[1]
    #         else:
    #             label_str += ' [O] ' + triplet[1]
    #             label_str += ' [A] ' + triplet[0]
    #     label_strs.append(label_str)

    # return ", [SSEP] ".join(label_strs)
    # for triplet in all_tri:
    #     label_str = ''
    #     if element1 == '[A]':
    #         label_str += '[A] ' + triplet[0]
    #         if element2 == '[O]':
    #             label_str += ' [O] ' + triplet[1]
    #             label_str += ' [S] ' + triplet[2]
    #         else:
    #             label_str += ' [S] ' + triplet[2]
    #             label_str += ' [O] ' + triplet[1]
    #     elif element1 == '[O]':
    #         label_str += '[O] ' + triplet[1]
    #         if element2 == '[A]':
    #             label_str += ' [A] ' + triplet[0]
    #             label_str += ' [S] ' + triplet[2]
    #         else:
    #             label_str += ' [S] ' + triplet[2]
    #             label_str += ' [A] ' + triplet[0]
    #     else:
    #         label_str += '[S] ' + triplet[2]
    #         if element2 == '[A]':
    #             label_str += ' [A] ' + triplet[0]
    #             label_str += ' [O] ' + triplet[1]
    #         else:
    #             label_str += ' [O] ' + triplet[1]
    #             label_str += ' [A] ' + triplet[0]
    #     label_strs.append(label_str)
    return " [SSEP] ".join(label_strs)


def get_input_label_position(words, target_indices, tokenizer, task):
    data = {}
    words = words + ["</s>"]
    s_to_t, cur_index = defaultdict(list), 0
    specific_tokens, specific_ids = [], []
    """
    将句子中的每个单词都转化为token和对应的token_id，并且存储在list中
    """
    for i in range(len(words)):
        # specific_token是字符串组成的列表
        specific_token = tokenizer.tokenize(words[i])
        specific_id = tokenizer.convert_tokens_to_ids(specific_token)
        specific_tokens.append(specific_token)
        specific_ids.append(specific_id)
        # 对序列号依次进行编号
        s_to_t[i] = [c for c in range(cur_index, cur_index + len(specific_token))]
        cur_index += len(specific_token)
    # lens = list(map(len, specific_tokens))
    # 将嵌套列表变成一个一维列表
    _specific_tokens = list(chain(*specific_tokens))
    _specific_ids = tokenizer.convert_tokens_to_ids(_specific_tokens)
    assert _specific_ids == list(chain(*specific_ids))

    aspect_label, opinion_label = [], []
    cum_aspect_label = [0] * len(words)
    cum_opinion_label = [0] * len(words)

    # BIO tagging scheme
    for triplet in target_indices:
        opinion_target_index = 1
        if task in ['asqp', 'acos']:
            opinion_target_index = 3
        # aspect
        cur_aspect_label = [0] * len(words)
        if triplet[0] != 'null':
            a_st, a_ed = triplet[0][0], triplet[0][-1]
            cur_aspect_label[a_st] = 2
            cum_aspect_label[a_st] = 2
            for i in range(a_st + 1, a_ed + 1):
                cur_aspect_label[i] = 1
                cum_aspect_label[i] = 1
        aspect_label.append(cur_aspect_label)
        # opinion
        cur_opinion_label = [0] * len(words)
        if triplet[opinion_target_index] != 'null':
            o_st, o_ed = triplet[opinion_target_index][0], triplet[opinion_target_index][-1]
            cur_opinion_label[o_st] = 2
            cum_opinion_label[o_st] = 2
            for i in range(o_st + 1, o_ed + 1):
                cur_opinion_label[i] = 1
                cum_opinion_label[i] = 1
        opinion_label.append(cur_opinion_label)
    data['pack_ids'] = specific_ids

    """
    将 _specific_ids 列表转换为 PyTorch 的 LongTensor 类型
    并使用 unsqueeze(0) 方法在第一个维度上添加了一个维度，从而将其转换为形状为 (1, length) 的张量，
    其中 length 是 _specific_ids 列表的长度。这通常是为了与模型的输入格式相匹配，
    因为许多模型要求输入是一个 batch 的数据，所以在最前面添加了一个维度来表示 batch size。
    """
    data['input_ids'] = torch.LongTensor(_specific_ids).unsqueeze(0)
    data['attention_mask'] = torch.LongTensor([1] * len(_specific_ids)).unsqueeze(0)
    data['aspect_label'] = torch.LongTensor(aspect_label)
    data['opinion_label'] = torch.LongTensor(opinion_label)

    """
    这段代码创建了一个二维列表 word_matrix，其行数等于 words 列表的长度，列数等于 _specific_tokens 列表的长度。
    每行代表一个单词在 _specific_tokens 中的出现情况，如果单词在 _specific_tokens 中出现，则对应位置的值为1，否则为0。
    """
    word_matrix = []
    for i in range(len(words)):
        row = [0] * len(_specific_tokens)
        for j in s_to_t[i]:
            row[j] = 1
        word_matrix.append(row)

    data['word_index'] = torch.LongTensor(word_matrix)
    data['word_mask'] = torch.LongTensor([1] * len(words))
    return data


def get_target_marker_position(target_seq, tokenizer):
    """
    data包含了input_ids和attention_mask
    input_ids：目标序列的tokenized表示，用整数表示每个token在词汇表中的索引
    attention_mask：一个与input_ids形状相同的张量，表示哪些位置需要注意，哪些位置是填充的
    """
    data = tokenizer(target_seq, return_tensors='pt')
    sentiment_polarity = {'positive': 0, 'negative': 1, 'neutral': 2}
    sentiment_marker = []
    # shape[-1]表示张量的最后一个维度的大小
    target_seq_len = data['input_ids'].shape[-1]
    marker_position = torch.zeros((target_seq_len,), dtype=torch.long)
    marker_names = {'[A]': 1, '[O]': 2, '[S]': 3, '[C]': 4}
    for index, item in enumerate(target_seq.split(' ')):
        if item == '[S]' and target_seq.split(' ')[index + 1] == 'positive':
            sentiment_marker.append(sentiment_polarity['positive'])
        if item == '[S]' and target_seq.split(' ')[index + 1] == 'negative':
            sentiment_marker.append(sentiment_polarity['negative'])
        if item == '[S]' and target_seq.split(' ')[index + 1] == 'neutral':
            sentiment_marker.append(sentiment_polarity['neutral'])
    sentiment_marker = torch.tensor(sentiment_marker).unsqueeze(0)
    data['sentiment_marker'] = sentiment_marker
    # sep_seq这个张量表示一个序列中的分隔符 token，这里的分隔符就是冒号”：“，10代表着冒号在词表中的序号
    # sep_seq = torch.tensor([908] * target_seq_len, dtype=torch.long)
    # sep_t是一个bool张量，ep函数来比较data['input_ids']和sep_seq对应位置的值是否一样，是则返回True，否则返回False，同时将sep_t在维度1上进行向左平移
    # sep_t = data['input_ids'].eq(sep_seq).roll(-1, dims=1)
    for marker_name, val in marker_names.items():
        marker_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(marker_name))[1]
        marker_seq = torch.tensor([marker_id] * target_seq_len, dtype=torch.long)
        t = data['input_ids'].eq(marker_seq)
        # 它根据给定的条件张量 t & sep_t 中的值，在两个张量 val 和 marker_position 之间选择。
        # 如果条件张量中的值为 True，则选择 val 中对应位置的值，否则选择 marker_position 中对应位置的值。
        marker_position = torch.where(t, val, marker_position)
    data['marker_position'] = marker_position
    return data


# 定义自定义的比较函数
def custom_sort(x, priority_order):
    # print(x)
    # print(x[3])
    # print(type(x[3]))
    if len(x) < 4:
        return (0, 0)
    # 提取子列表的第一个和第四个元素
    A = x[0]
    O = x[3]
    # print(type(O))
    # 如果第一个元素是字符串"null"，则将其替换为最小值
    if A == "null":
        A_out = float('-inf')
    else:
        A_out = A[-1]

    # 如果第四个元素是字符串"null"，则将其替换为最小值
    # if O == "null":
    if isinstance(O, list):
        # print('OOOOOOOOOOOOO')
        # print(O)
        # print(type(O))
        O_out = O[-1]
    else:
        O_out = float('-inf')
    # 根据优先次序参数决定排序的优先次序
    if priority_order == "AO":
        return (A_out, O_out)
    elif priority_order == "OA":
        return (O_out, A_out)


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_path, opt):
        super(ABSADataset, self).__init__()
        self.opt = opt
        self.tokenizer = tokenizer
        self.task_name = opt.task_name + 'unified'
        self.data_name = opt.dataset
        self.data_path = data_path
        self.data_type = 'train'
        self.top_k = opt.top_k
        self.multi_task = opt.multi_task
        if opt.full_supervise:
            # 返回没经过修改的tasks, datas, sents, labels
            # self._build_examples()
            self.all_inputs, self.all_targets = get_transformed_io_unified(
                self.data_path, self.task_name, self.data_name, self.data_type,
                self.top_k, self.opt)

        # else:
        #     self.all_inputs, self.all_targets = read_shot_ratio_from_file(data_path)
        self.all_additions = {}
        """
        遍历了 tokenizer 的词汇表k是单词，v是对应的id，对于以逗号结尾且长度大于等于 2 的标记，
        将其除去逗号后转换为 ID，并将该 ID 作为键，原始标记作为值存储在 self.all_additions 字典中
        """
        for k, v in tokenizer.get_vocab().items():
            if k[-1] == ',' and len(k) >= 2:
                x = tokenizer.convert_tokens_to_ids(k[:-1])
                self.all_additions[x] = v
        # if self.opt.data_format == 'A':
        #     self.marker_orders = ['aspect'] * len(self.all_inputs)
        # if self.opt.data_format == 'O':
        #     self.marker_orders = ['opinion'] * len(self.all_inputs)
        # if self.opt.data_format == 'AO':
        #     self.marker_orders = ['aspect'] * len(self.all_inputs) + ['opinion'] * len(self.all_inputs)
        #     self.all_inputs = self.all_inputs + self.all_inputs
        #     self.all_targets = self.all_targets + self.all_targets

    def __getitem__(self, index):
        input_seq, target_seq = self.all_inputs[index].copy(), self.all_targets[index].copy()

        task = input_seq[0]
        target_aspect_index = 0
        target_opinion_index = 1
        marker_order = input_seq[-4:]
        if '[C]' not in marker_order:
            marker_order = marker_order[-3:]
        aspect_index = marker_order.index('[A]')
        opinion_index = marker_order.index('[O]')
        if len(target_seq) > 1:
            if task in ['asqp', 'acos']:
                if aspect_index < opinion_index:
                    target_seq = sorted(target_seq, key=lambda x: custom_sort(x, priority_order="AO"))
                else:
                    target_seq = sorted(target_seq, key=lambda x: custom_sort(x, priority_order="OA"))
            else:
                if aspect_index < opinion_index:
                    target_seq.sort(key=lambda x: (convert_to_int(x[0][-1]), convert_to_int(x[1][-1])))
                else:
                    target_seq.sort(key=lambda x: (convert_to_int(x[1][-1]), convert_to_int(x[0][-1])))

        add_len = 4
        # 根据情感元素的顺序来调整四元组的依次出现的顺序
        if task in ['asqp', 'acos']:
            target_opinion_index = 3
        for i in range(len(target_seq)):
            if target_seq[i][target_aspect_index] != 'null':
                a = [x + add_len for x in target_seq[i][target_aspect_index]]
            else:
                a = target_seq[i][target_aspect_index]
            if target_seq[i][target_opinion_index] != 'null':
                b = [x + add_len for x in target_seq[i][target_opinion_index]]
            else:
                b = target_seq[i][target_opinion_index]
            s = target_seq[i][2]
            if task in ['asqp', 'acos']:
                c = target_seq[i][1]
                target_seq[i] = (a, c, s, b)
            else:
                target_seq[i] = (a, b, s)
        target_copy = list(target_seq)
        # 将list四元组变成字符串四元组
        target_seq = ABSA_format(input_seq, target_seq, task)

        source = get_input_label_position(input_seq, target_copy, self.tokenizer, task)
        target = get_target_marker_position(target_seq, self.tokenizer)
        # print(f"target: {torch.sum(target['marker_position'].squeeze(0).eq(1))}")
        # print(f"source: {source['aspect_label'].shape[0]}")
        assert torch.sum(target['marker_position'].squeeze(0).eq(1)) == source['aspect_label'].shape[0]

        next_ids = defaultdict(list)
        # 将tensor转化为list并且选择第零维的内容
        input_ids = source['input_ids'].tolist()[0]
        next_ids[1] = []  # last pad token
        next_ids[-1] = []  # addition token
        next_ids[0] = []
        """
        遍历sent转成的token列表，形成next_ids字典，键是当前的tokenid，值是下一个的tokenid，如果遍历到句子的最后一个词，则它的下一个tokenid就是"</s>"的id，
        """
        for i in range(7, len(input_ids) - 4):
            cur = input_ids[i]
            # ne = None if i == len(input_ids) - 1 else input_ids[i + 1]
            ne = 1 if i == len(input_ids) - 5 else input_ids[i + 1]
            # ne = None if i == len(input_ids) - 5 else input_ids[i + 1]
            # 如果存在某些token有与其一致的token带了逗号，则用带逗号的token替代不带逗号的token
            # if ne in self.all_additions:
            #     next_ids[cur].append(self.all_additions[ne])
            #     next_ids[-1].append(self.all_additions[ne])
            if ne:
                next_ids[cur].append(ne)
        # 每个词都会经过tokenizer转化成token list，在每个token list里的最后一个元素后添加逗号，逗号的tokenid是6
        for cur_ids in source['pack_ids'][4:-4]:
            if len(cur_ids) == 1:
                next_ids[cur_ids[0]].append(6)
            else:
                next_ids[cur_ids[-1]].append(6)
        next_ids[1].append(6)
        next_ids = dict(next_ids)
        """
        input_ids：input_ids是模型输入的token ID序列，它表示将输入文本转换为模型可以理解的数字形式。
        这个数字序列是根据预先构建的词汇表，将文本中的每个单词或子词映射到唯一的整数 ID。

        attention_mask：attention_mask 是用于指示模型在处理输入时应该关注哪些位置的一个张量。
        在自然语言处理任务中，输入序列通常是变长的，但神经网络需要固定大小的输入。
        因此，为了处理不同长度的输入序列，通常会使用填充（padding）将较短的序列填充到相同的长度。
        attention_mask 的作用就是告诉模型哪些位置是真实的输入，哪些位置是填充的无效位置。
        它是一个与输入序列相同长度的二元张量，其中真实的输入位置对应的元素值为 1，填充位置对应的元素值为 0。
        这样，模型在处理输入序列时就可以根据 attention_mask 来忽略填充位置的信息，只关注真实的输入内容。

        input_seq：是指输入序列，通常是经过预处理后的文本序列（文本形式），可能包含了特殊的标记或前缀。

        aspect_label：方面词的BIO序列，方面词的开始标记为2，其他部分标记为1，不属于方面词的标记为0

        opinion_label：观点词的BIO序列，观点词的开始标记为2，其他部分标记为1，不属于观点词的标记为0

        marker_position：marker序列，将句子中的aspect标记为1，opinion标记为2，sentiment标记为3，其余的标记为0

        word_index：一个张量，用于表示输入序列中的单词在词汇表中的索引

        word_mask：是一个张量，用于表示输入序列中哪些位置是实际的单词，哪些位置是填充或者特殊标记。
        通常，这个张量的形状与输入序列的形状相同，每个位置的值为1表示是实际的单词，值为0表示是填充或者特殊标记。

        next_ids：是一个字典，其键值对表示了当前单词的ID和其下一个可能的单词的ID。具体来说，它是一个将当前单词ID映射到下一个可能单词ID的字典。

        marker_order：表示了当前样本中标记的顺序类型，可能的取值为 'aspect'、'opinion' 或者 'aspect_opinion'。
        它指示了在当前样本中，标记的顺序类型是什么，以便在数据处理和模型训练过程中进行相应的操作。
        """
        return {
            "index": index,
            "input_ids": source['input_ids'].squeeze(0),
            "attention_mask": source['attention_mask'].squeeze(0),
            "labels": target['input_ids'].squeeze(0),
            "decoder_attention_mask": target['attention_mask'].squeeze(0),
            "input_seq": input_seq,
            "target_seq": self.all_targets[index],
            "aspect_label": source['aspect_label'],
            "opinion_label": source['opinion_label'],
            "marker_position": target['marker_position'],
            "sentiment_marker": target['sentiment_marker'],
            "word_index": source["word_index"],
            "word_mask": source["word_mask"],
            "next_ids": next_ids
            # "marker_order": marker_order
        }
        # return {
        #     "index": index,
        #     "input_seq": input_seq,
        #     "target_seq": target_seq,
        #     # "all_inputs": self.all_inputs,
        #     # "all_targets": self.all_targets,
        #     # "new_sents": new_sents,
        #     # "new_targets": new_targets
        #     "source": source,
        #     "target": target,
        # }

    # def __getitem__(self, index):
    #     return {
    #         "index": index,
    #         "all_inputs": self.all_inputs,
    #         "all_targets": self.all_targets
    #     }

    def __len__(self):
        return len(self.all_inputs)

    # def _build_examples(self):
    #     print(f"top_k: {self.top_k}")
    #     self.all_inputs, self.all_targets = get_transformed_io_unified(
    #         self.data_path, self.task_name, self.data_name, self.data_type,
    #         self.top_k, self.opt)


def convert_to_int(value):
    # print(value)
    return -1 if value == 'l' else int(value)


def collate_func_train(batch):
    # ao_data, oa_data = {}, {}
    # ao_batch = [batch[i] for i in range(len(batch)) if batch[i]['marker_order'] == 'aspect']
    # oa_batch = [batch[i] for i in range(len(batch)) if batch[i]['marker_order'] == 'opinion']

    # pad_batch_data(ao_batch, ao_data)
    # pad_batch_data(oa_batch, oa_data)
    data = {}
    batch = [batch[i] for i in range(len(batch))]
    sentiment_marker = torch.tensor([])
    for batch_item in batch:
        if "sentiment_marker" in batch_item:
            sentiment_marker = torch.cat((sentiment_marker, batch_item['sentiment_marker']), dim=1)

    pad_batch_data(batch, data)
    # print(f"ao_data: {ao_data}")
    # print(f"oa_data: {oa_data}")
    data['sentiment_marker'] = sentiment_marker
    return {"data": data}


# def collate_func_eval(batch):
#     data = {}
#     pad_batch_data(batch, data)
#     return data

def collate_func_eval(batch):
    data = {}
    pad_batch_data(batch, data)
    sentiment_marker = torch.tensor([])
    for batch_item in batch:
        if "sentiment_marker" in batch_item:
            sentiment_marker = torch.cat((sentiment_marker, batch_item['sentiment_marker']), dim=1)
    data['sentiment_marker'] = sentiment_marker
    return data


def pad_batch_data(cur_batch, cur_data):
    if len(cur_batch) == 0:
        return
    for k, v in cur_batch[0].items():
        if k in ['word_index']:
            cur_data[k] = padded_stack([s[k] for s in cur_batch])
            continue
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 1:
                cur_data[k] = pad_sequence([x[k].squeeze(0) for x in cur_batch], batch_first=True)
            else:
                rows = [list(map(lambda c: c.squeeze(0), torch.split(x[k], 1, dim=0))) for x in cur_batch]
                cur_data[k] = pad_sequence(list(chain(*rows)), batch_first=True)
        else:
            cur_data[k] = [x[k] for x in cur_batch]


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


class ASTESampler(Sampler):
    def __init__(self, data_source, target_format):
        super().__init__(data_source)
        self.target_format = target_format
        self.data_source = data_source
        # self.data_range = []
        # if target_format == 'AO':
        #     length = len(data_source) // 2
        #     a = [c for c in range(length)]
        #     o = [c for c in range(length, 2 * length)]
        #     for i in range(length):
        #         """
        #         则假设数据源中前一半是 'A' 类型的数据，后一半是 'O' 类型的数据，然后将索引按照 'A'-'O' 的顺序配对
        #
        #         将 'A' 和 'O' 配对可能是因为在训练过程中需要同时使用 'A' 和 'O' 类型的数据进行模型训练，以便模型能够学习到它们之间的关联关系。
        #         通过将 'A' 和 'O' 数据按顺序配对，可以确保每个 batch 中同时包含 'A' 和 'O' 类型的数据，
        #         从而使模型在每个 batch 中都能够接触到 'A' 和 'O' 之间的关系，有助于提高模型的训练效果。
        #         """
        #         self.data_range.append([a[i], o[i]])
        # else:
        #     for i in range(len(data_source)):
        #         self.data_range.append(i)
        length = len(data_source)
        inputs_index = [c for c in range(length)]
        group_size = 5
        self.data_range = [inputs_index[i:i + group_size] for i in range(0, len(inputs_index), group_size)]

    def __iter__(self):
        np.random.shuffle(self.data_range)
        if isinstance(self.data_range[0], list):
            self.data_range = list(chain(*self.data_range))
        return iter(self.data_range)

    def __len__(self):
        return len(self.data_source)
