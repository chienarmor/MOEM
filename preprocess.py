import os
import random
import sys

sys.path.append(".")

from collections import Counter
from utils.data_utils import parse_aste_tuple

join = os.path.join


def parse_tuple(sentence, lists, task):
    words = sentence.split()
    if task in ['acos', 'asqp']:
        for quadruple_index, quadruple in enumerate(lists):
            A = quadruple[0]
            O = quadruple[3]
            if A != 'null':
                A_list = find_word_indices(words, A)
                lists[quadruple_index][0] = A_list
            if O != 'null':
                # print(f"words: {words}")
                # print(f"O: {O}")
                O_list = find_word_indices(words, O)
                lists[quadruple_index][3] = O_list
    if task in ['tasd']:
        lists = [list(tup) for tup in lists]
        for triple_index, triple in enumerate(lists):
            A = triple[0]
            if A != 'null':
                # print(f"words: {words}")
                # print(f"A: {A}")
                A_list = find_word_indices(words, A)
                lists[triple_index][0] = A_list
    return lists


def find_word_indices(sent, element):
    element = element.split()
    element_list = []
    for item in element:
        element_list.append(sent.index(item))
    return element_list


def process(data_folder, tasks, out_dir, data_name=None):
    """
    1. Aggregate all train, dev, and test sets for the tasks acos/asqp/aste/tasd.
    2. Remove data contamination: delete the test set data that exists in the train/dev sets.
    3. Output data.txt
    Data format: (task, data, words, tuples)
    """
    train_data = []
    dev_data = []
    test_data = []
    # merge all data
    for task in tasks:
        task_path = join(data_folder, task)
        print("task:", task_path)
        if data_name is not None:
            data_path = join(task_path, data_name)
            print("data:", data_path)
            # acos data_name
            for split in ["train", "dev", "test"]:
                with open(join(data_path, "{}.txt".format(split)),
                          'r',
                          encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip().lower()
                        if line != '':
                            words, tuples = line.split('####')
                        # parse aste
                        # if task == "aste":
                        #     aste_tuples = []
                        #     for _tuple in eval(tuples):
                        #         parsed_tuple = parse_aste_tuple(
                        #             _tuple, words.split())
                        #         aste_tuples.append(parsed_tuple)
                        #     tuples = str(aste_tuples)

                        # output
                        # print(f"words: {words}")
                        # print(f"tuples: {tuples}")
                        if task != 'aste':
                            tuples = eval(tuples)
                            tuples = str(parse_tuple(words, tuples, task))
                        if split == "test":
                            test_data.append((task, data_name, words, tuples))
                        elif split == "train":
                            train_data.append((task, data_name, words, tuples))
                        else:
                            dev_data.append((task, data_name, words, tuples))
        else:
            for data_name in os.listdir(task_path):
                data_path = join(task_path, data_name)
                print("data:", data_path)
                # acos data_name
                for split in ["train", "dev", "test"]:
                    with open(join(data_path, "{}.txt".format(split)),
                              'r',
                              encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip().lower()
                            if line != '':
                                words, tuples = line.split('####')
                            # parse aste
                            # if task == "aste":
                            #     aste_tuples = []
                            #     for _tuple in eval(tuples):
                            #         parsed_tuple = parse_aste_tuple(
                            #             _tuple, words.split())
                            #         aste_tuples.append(parsed_tuple)
                            #     tuples = str(aste_tuples)

                            # output
                            # print(f"words: {words}")
                            # print(f"tuples: {tuples}")
                            if task != 'aste':
                                tuples = eval(tuples)
                                tuples = str(parse_tuple(words, tuples, task))
                            if split == "test":
                                test_data.append((task, data_name, words, tuples))
                            elif split == "train":
                                train_data.append((task, data_name, words, tuples))
                            else:
                                dev_data.append((task, data_name, words, tuples))
    # print(train_data[0])
    # print(dev_data[0])
    # print(test_data[0])
    # # remove inputs in test set
    # test_inputs = set()
    # for _, _, words, _ in test_data:
    #     test_inputs.add(words.replace(" ", ""))
    # print(list(test_inputs)[0])
    # train_data_safe = []
    # for item in train_data:
    #     if item[2].replace(" ", "") not in test_inputs:
    #         train_data_safe.append(item)
    # print(train_data_safe[0])
    # print("test inputs size:", len(test_inputs))
    # print("train data size (before remove test):", len(train_data))
    # print("train data size (after remove test):", len(train_data_safe))
    #
    # # dedup
    # random.seed(0)
    # random.shuffle(train_data_safe)
    # train_data_dedup = []
    # train_pairs = set()
    # for item in train_data_safe:
    #     pair = (item[2] + item[3]).replace(" ", "")
    #     if pair not in train_pairs:
    #         train_pairs.add(pair)
    #         train_data_dedup.append(item)
    #
    # print("train data size (dedup):", len(train_data_dedup))
    # print(train_data_dedup[0])
    # # stats
    # task_list = []
    # data_list = []
    # for task, data_name, _, _ in train_data_dedup:
    #     task_list.append(task)
    #     data_list.append(data_name)
    # print("Tasks counts:", Counter(task_list))
    # print("Data counts:", Counter(data_list))

    # output
    for seed in [5, 10, 15, 20, 25, 30, 35, 40, 45, 3407]:
        os.makedirs(out_dir + "/seed{}".format(seed), exist_ok=True)
        random.seed(seed)
        random.shuffle(train_data)
        # idx = int(len(train_data_dedup) * 0.9)
        # train_set = train_data_dedup[:idx]
        # dev_set = train_data_dedup[idx:]

        # sort
        # train_data = sorted(train_data, key=lambda x: x[2])
        dev_data = sorted(dev_data, key=lambda x: x[2])
        test_data = sorted(test_data, key=lambda x: x[2])

        with open(out_dir + "/seed{}/train.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            train_len_split = len(train_data)//10
            # train_len_split = 30
            for item in train_data[:train_len_split]:
                fp.write("{}\t{}\t{}####{}\n".format(*item))

        with open(out_dir + "/seed{}/dev.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in dev_data:
                fp.write("{}\t{}\t{}####{}\n".format(*item))

        with open(out_dir + "/seed{}/test.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in test_data:
                fp.write("{}\t{}\t{}####{}\n".format(*item))


if __name__ == "__main__":
    tasks = ["aste"]
    # tasks = ["aste", "acos", "asqp"]
    # process("data", tasks, "data/unified/")
    data_names = ['laptop14', 'rest14', 'rest15','rest16']
    # data_name = "rest15"
    # process("data", tasks, "data/aste/unified")
    for data_name in data_names:
        process("data", tasks, f"data/aste/{data_name}-ratio/0.1", data_name)
