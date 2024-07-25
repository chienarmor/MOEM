import sys
import os
import pdb
from dataclasses import dataclass, field
from typing import Optional
import transformers
import torch
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    T5Config,
    T5Tokenizer,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from models import T5ForConditionalGeneration, MarkerT5
import logging
from utils.data_utils import ABSADataset
from marker_trainer import MarkerSeq2SeqTrainer
from utils.eval_utils import parse_and_score

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="../../PretrainModel/t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(
        default='unified', metadata={"help": "acos/asqp/aste/tasd/unified "}
    )
    dataset: str = field(
        default='laptop14', metadata={"help": "laptop14/rest14/rest15/rest16 "}
    )
    data_format: str = field(
        default="A", metadata={"help": "A/O/AO"}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of train data"}
    )
    valid_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of valid data"},
    )
    test_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of test data"},
    )
    max_length: int = field(
        default=300, metadata={"help": "The max padding length of source text and target text"}
    )
    num_beams: int = field(
        default=1, metadata={"help": "greedy search"}
    )
    shot_ratio_index: Optional[str] = field(
        default="-1[+]-1[+]1", metadata={"help": "1[+]-1[+]1->1-shot"}
    )
    lowercase: Optional[bool] = field(
        default=None, metadata={"help": "lowercase sentences"}
    )
    multi_path: Optional[bool] = field(
        default=None, metadata={"help": "multi_path"}
    )
    single_view_type: str = field(
        default='rank', metadata={"help": "the method of selecting the views of all choices"}
    )
    sort_label: Optional[bool] = field(
        default=None, metadata={"help": "sort tuple by order of appearance"}
    )
    ctrl_token: str = field(
        default='post', metadata={"help": "combine sentence and orders"})
    multi_task: Optional[bool] = field(
        default=None, metadata={"help": "multi_task"}
    )
    top_k: int = field(
        default=0, metadata={"help": "choose an order"}
    )

    def __post_init__(self):
        if self.dataset is None and self.train_path is None and self.valid_path is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        shot, ratio, index = self.shot_ratio_index.split("[+]")
        shot, ratio, index = int(shot), float(ratio), int(index)
        assert shot in [-1, 1, 5, 10] and ratio in [-1, 0.01, 0.05, 0.1]
        # self.full_supervise = True if shot == -1 and ratio == -1 else False
        self.full_supervise = True
        name_mapping = {"laptop14": "14lap", "rest14": "14res", "rest15": "15res", "rest16": "16res"}
        if self.multi_task:
            if shot != -1:
                # self.train_path = f'./data/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/train.json'
                # self.valid_path = f'./data/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/val.json'
                # self.test_path = f'./dataa/ASTE/{name_mapping[self.dataset]}_shot/seed{index}/{shot}shot/test.json'
                self.train_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/train.txt'
                self.valid_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/dev.txt'
                self.test_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/test.txt'
            elif ratio != -1:
                # self.train_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/train.json'
                # self.valid_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/val.json'
                # self.test_path = f'./data/ASTE/{name_mapping[self.dataset]}_ratio/seed{index}/{ratio}/test.json'
                self.train_path = f'./data_my/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/train.txt'
                self.valid_path = f'./data_my/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/dev.txt'
                self.test_path = f'./data_my/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/test.txt'
            else:
                # self.train_path = f'./data/{self.dataset}/seed5/train.txt'
                # self.valid_path = f'./data/{self.dataset}/seed5/dev.txt'
                # self.test_path = f'./data/{self.dataset}/seed5/test.txt'
                # self.test_path = f'./data/aste/rest14/test.txt'
                self.train_path = f'./data_my/{self.task_name}/{self.dataset}/seed3407/train.txt'
                self.valid_path = f'./data_my/{self.task_name}/{self.dataset}/seed3407/dev.txt'
                self.test_path = f'./data_my/{self.task_name}/{self.dataset}/seed3407/test.txt'
                # self.test_path = f'./data_my/aste/unified/rest15.txt'
        else:
            self.train_path = f'./data_my/{self.dataset}/seed3407/train.txt'
            self.valid_path = f'./data_my/{self.dataset}/seed3407/dev.txt'
            self.test_path = f'./data_my/{self.dataset}/seed3407/test.txt'

        self.source_aspect_prefix = ["aspect", "first:"]
        self.source_opinion_prefix = ["opinion", "first:"]
        self.prefix_word_length = 2  # ["aspect", "first:"]
        self.prefix_token_length = 3  # ["aspect", "first", ":"]

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


@dataclass
class MarkerSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments for our model in training procedure
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    use_marker: bool = field(default=False, metadata={"help": "Whether to use marker"})
    marker_type: Optional[str] = field(default='AO', metadata={"help": "marker_type A/O"})
    alpha: float = field(default=0.5, metadata={"help": "adjust the loss weight of ao_template and oa_template"})
    progressive_feature: bool = field(default=False, metadata={"help": "Whether to use progressive feature"})


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MarkerSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        # print("sys.argv[1]: ",sys.argv[1])
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print(model_args, data_args, training_args, sep='\n')
    # model_args, data_args, training_args = parser.parse_json_file(json_file="./args.json")
    # print(model_args, data_args, training_args, sep='\n')
    # print(f"model_args: {model_args}")
    # print(f"data_args: {data_args}")
    # print(f"training_args: {training_args}")

    # if data_args.data_format == "AO":
    #     training_args.per_device_train_batch_size *= 2

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # ipdb.set_trace()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # pass args
    config.max_length = data_args.max_length
    config.num_beams = data_args.num_beams

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # tokenizer.add_tokens(['[SSEP]'])  # tokenid 32100
    # tokenizer.add_tokens(['[A]'])  # tokenid 32101
    # tokenizer.add_tokens(['[O]'])  # tokenid 32102
    # tokenizer.add_tokens(['[S]'])  # tokenid 32103
    # tokenizer.add_tokens(['[SSEP]'])
    # test_str = "[O] unlimited [A] sushi [S] great [SSEP] [O] high [A] quality [S] great [SSEP] [O] best [A] sushi places [S] great"
    # test_result = tokenizer(test_str, return_tensors='pt')
    # print(f"test_result: {test_result}")
    # target_seq_len = test_result['input_ids'].shape[-1]
    # marker_position = torch.zeros((target_seq_len,), dtype=torch.long)
    # marker_names = {'[A]': 1, '[O]': 2, '[S]': 3}
    #
    # for marker_name, val in marker_names.items():
    #     marker_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(marker_name))[0]
    #     print(f"the marker_id of {marker_name}: {marker_id}")
    #     marker_seq = torch.tensor([marker_id] * target_seq_len, dtype=torch.long)
    #     print(f"the marker_seq of {marker_name}: {marker_seq}")
    #     t = test_result['input_ids'].eq(marker_seq)
    #     print(t)
    #     # 它根据给定的条件张量 t & sep_t 中的值，在两个张量 val 和 marker_position 之间选择。
    #     # 如果条件张量中的值为 True，则选择 val 中对应位置的值，否则选择 marker_position 中对应位置的值。
    #     # marker_position = torch.where(t, val, marker_position)
    #     marker_position = torch.where(t, val, marker_position)
    #     print(f"marker_position: {marker_position}")
    # test_result['marker_position'] = marker_position
    pretrain_model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    pretrain_model.resize_token_embeddings(len(tokenizer))
    model = MarkerT5(
        args=data_args,
        tokenizer=tokenizer,
        t5_model=pretrain_model,
        t5_config=config,
        use_marker=training_args.use_marker,
        marker_type=training_args.marker_type,
        progressive_feature=training_args.progressive_feature
    )
    if not training_args.do_train:
        logger.info("load checkpoint of Marker-T5 !")
        # model.load_state_dict(torch.load(f"{training_args.output_dir}/checkpoint-6832/pytorch_model.bin"))
        model.load_state_dict(
            torch.load(f"outputs/aste/unified_seed3407/top1/checkpoint-1460/pytorch_model.bin"))
        # model.load_state_dict(
        #     torch.load(f"outputs_models/AO-fpsl/unified_5/checkpoint-7120/pytorch_model.bin"))
    # print(data_args.train_path)
    train_dataset = ABSADataset(tokenizer=tokenizer, data_path=data_args.train_path, opt=data_args)
    # pdb.set_trace()
    eval_dataset = ABSADataset(tokenizer=tokenizer, data_path=data_args.valid_path, opt=data_args)
    test_dataset = ABSADataset(tokenizer=tokenizer, data_path=data_args.test_path, opt=data_args)

    # print("Here are examples (from the train set):")
    # for i in range(0, min(1, 10)):
    # sample = train_dataset[i]
    #     print(f"Sample {i + 1} - Index: {sample['index']}")
    #
    #     print("Input Sequence:", sample['input_seq'])
    #     print("Target Sequence:", sample['target_seq'])
    #     print("Aspect Label:", sample['aspect_label'])
    #     print("Opinion Label:", sample['opinion_label'])
    #     print("Marker Position:", sample['marker_position'])
    #     print("Word Index:", sample['word_index'])
    #     print("Word Mask:", sample['word_mask'])
    #     print("Next IDs:", sample['next_ids'])
    #     # print("Marker Order:", sample['marker_order'])
    #     print("input_ids:", sample['input_ids'])
    #     print("labels:", sample['labels'])
    #     # print("ao_data:", sample['ao_data'])
    #     # print("tasks:", sample['tasks'][i])
    #     # print("datas:", sample['datas'][i])
    #     # print("input_seq:", sample['input_seq'])
    #     print("target_seq:", sample['target_seq'])
    #     # print("source:", sample['source'])
    #     # print("target:", sample['target'])
    #     # for new_sent in sample['new_sents']:
    #     #     print(new_sent)
    #     # for new_target in sample['new_targets']:
    #     #     print(new_target)
    #     # print("all_inputs:", sample['all_inputs'][i])
    #     # print("all_targets:", sample['all_targets'][i])
    #     # print("new_sents:", sample['new_sents'])
    #     # print("new_targets:", sample['new_targets'])
    #     # sents = sample['all_inputs']
    #     # for sent in sents:
    #     #     if ']' in sent:
    #     #         print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
    #     print("\n")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # if labels[0][-1] == -100:
        #     labels = [sublist[:-1] for sublist in labels]
        # if labels[0][-1] == -100:
        #     labels = labels[:, :399]
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # print(preds[:5])
        # print(f"pred: {preds}")
        # pdb.set_trace()
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # result = parse_and_score(preds, labels, data_args.data_format)
        result = parse_and_score(preds, labels, data_args)
        return result

    trainer = MarkerSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_format=data_args.data_format,
        tokenizer=tokenizer,
        ignore_pad_token=data_args.ignore_pad_token_for_loss,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None)

    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

    if training_args.do_predict:
        logger.info(f"*** Test constraint decoding: {training_args.constraint_decoding}***")
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.max_length,
            num_beams=data_args.num_beams,
            constraint_decoding=training_args.constraint_decoding,
        )
        # with open("./slgm-rest14-46.txt", 'w', encoding='utf-8') as f:
        #     for idx, prediction in enumerate(test_results.predictions):
        #         decoded_text = tokenizer.decode(prediction, skip_special_tokens=True)
        #         # print(f"Example {idx + 1} Decoded Text: {decoded_text}")
        #         f.write(decoded_text + '\n')
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)
        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"{key} = {value}")
                    writer.write(f"{key} = {value}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
