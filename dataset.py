#!/usr/bin/python3
import logging

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset, TensorDataset, random_split
from torch import tensor, cat

from config import MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE


def generate_dataset(sentences, tokenizer, labels):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            return_attention_mask=True,  # Construct attn. masks.
            max_length=MAX_LEN,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_tensors='pt',  # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    # Convert the lists into tensors.
    input_ids = cat(input_ids, dim=0)
    attention_masks = cat(attention_masks, dim=0)
    token_type_ids = cat(token_type_ids, dim=0)
    labels=tensor(labels)

    dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)

    i = 0
    logging.info("\n ***  encode examples %s ***" % id)
    logging.info("sentence: %s" % sentences[i])
    logging.info("tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids[i])))
    logging.info("input_ids: %s" % input_ids[i])
    logging.info("token_type_ids: %s" % token_type_ids[i])
    logging.info("attention_mask: %s" % attention_masks[i])
    logging.info("label: %s " % labels[i])
    return dataset


def split_train_test_set(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Calculate the number of samples to include in each set.
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    logging.info('{:>5,} training samples'.format(train_size))
    logging.info('{:>5,} validation samples'.format(val_size))
    logging.info('{:>5,} test samples'.format(val_size))
    return train_dataset, val_dataset, test_dataset


# create batch
def create_dataloader(train_dataset, val_dataset):
    logging.info('Train Batch size: ', TRAIN_BATCH_SIZE)
    logging.info('Valid Batch size: ', VALID_BATCH_SIZE)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=TRAIN_BATCH_SIZE  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=VALID_BATCH_SIZE  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader
