#!/usr/bin/python3
from __future__ import print_function
import datetime
import logging
import random
import time

import numpy as np
import torch
from torch import cuda, manual_seed
from early_stop import EarlyStopping
from evaluation import multi_label_metrics, F1Score
import config


def train_model(model, train_dataloader, validation_dataloader, scheduler, optimizer, epochs, class_weights):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    manual_seed(seed_val)
    cuda.manual_seed_all(seed_val)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor(class_weights).to(device))

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    early_stopping = EarlyStopping(config.MODEL_SAVE_PATH)
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logging.info('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 20 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                logging.info(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # "logits"--the model outputs prior to activation.
            logits = model(input_ids=b_input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=b_input_mask,
                           # labels=b_labels,
                           # return_dict=True
                           )
            loss = criterion(logits, b_labels)
            # logits = result.logits
            # loss = result.loss

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        logging.info("")
        logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")
        logging.info("")
        logging.info("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                logits = model(input_ids=b_input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=b_input_mask,
                               # labels=b_labels,
                               # return_dict=True
                               )
                loss = criterion(logits, b_labels)
            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            thresh = 0.5
            if epoch_i == 0:
                f_score = F1Score(search_thresh=True)
                f_score(logits, b_labels)
                thresh = f_score.get_thresh()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            metrics = multi_label_metrics(logits, label_ids, threshold=thresh)
            total_eval_accuracy += metrics['f1']

            if step % 20 == 0 and not step == 0:
                logging.info('==== print every 40 steps =====\n')
                logging.info('==== output logits =====\n')
                logging.info(logits)
                logging.info('==== true labels =====\n')
                logging.info(label_ids)

                logging.info('==== evaluation =====')
                logging.info('support: %s' % str(metrics['support']))
                logging.info('subset acc: %s' % str(metrics['s_accuracy']))
                logging.info('hamming_score: %s' % str(metrics['hamming_score']))
                logging.info('precision: %s' % str(metrics['precision']))
                logging.info('recall: %s' % str(metrics['recall']))
                logging.info('f_score: %s' % str(metrics['f1']))

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. F1': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        # configure early stop
        logging.info('Early stopping check ...')
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info('Early stop now .')
            break

    print("")
    print("Training complete!")
    logging.info("")
    logging.info("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    logging.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return training_stats


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
