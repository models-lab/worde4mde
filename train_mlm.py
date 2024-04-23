from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, EarlyStoppingCallback


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.base_model) #, device_map="auto"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    dataset = load_dataset("json", data_files=args.parsed_dataset)["train"]
    splits = dataset.train_test_split(test_size=0.2, seed=123)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    splits_datasets = splits.map(tokenize_function, batched=True, num_proc=12,
                                 remove_columns=splits["train"].column_names)
    lm_datasets = splits_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    training_args = TrainingArguments(
        output_dir=args.checkpoint,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
        do_eval=True,
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        save_strategy="epoch",
        save_total_limit=2
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for training mlm')
    parser.add_argument('--parsed_dataset', default='modeling_corpus.jsonl')
    parser.add_argument('--checkpoint', default='roberta-modeling')
    parser.add_argument('--base_model', default='roberta-base')
    parser.add_argument('--block_size', default=512)
    args = parser.parse_args()
    main(args)
