import numpy as np
import torch
from scipy import stats as stats
from sklearn.metrics import f1_score
from torch import nn as nn

from my_functions import precision_k, print_num_on_tqdm, tqdm_with_num, print_multiple_metrics


def training(params, model, train_loader, optimizer, epoch=1, logger=None):
    device = params["device"]
    batch_total = params["train_batch_total"]
    loss_func = nn.BCELoss()

    model.train()
    losses = []

    # Show loss with tqdm
    with tqdm_with_num(train_loader, batch_total) as loader:
        loader.set_description("Training  ")

        # Batch Loop
        for idx, batch in enumerate(loader):
            # ---------------------- Main Process -----------------------
            data, target = (batch.text.to(device), batch.label.to(device))

            optimizer.zero_grad()

            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            loss = loss_func(outputs, target)

            loss.backward()
            optimizer.step()
            # -----------------------------------------------------------

            # Print training progress
            losses.append(loss.item())

            # Log batch loss if logger is available
            if logger is not None:
                logger.log_train_loss(epoch, idx, loss.item())

            if idx < batch_total - 1:
                print_num_on_tqdm(loader, loss)
            else:
                loss_epoch = np.mean(losses)
                print_num_on_tqdm(loader, loss_epoch, last=True)

                # Log average epoch loss
                if logger is not None:
                    logger.log_train_loss(epoch, avg_loss=loss_epoch)


def validating_testing(params, model, data_loader, epoch=1, is_valid=True, logger=None):
    device = params["device"]
    measure = params["measure"]
    doc_key = is_valid and "valid" or "test"
    batch_total = params[doc_key + "_batch_total"]

    model.eval()

    eval_epoch = 0.0
    target_all = np.empty((0, params["num_of_class"]), dtype=np.int8)
    eval_all = np.empty((0, params["num_of_class"]), dtype=np.float32)

    # For storing multiple metrics
    precision_values = {'p@1': 0.0, 'p@3': 0.0, 'p@5': 0.0}

    # Show p@k with tqdm
    with tqdm_with_num(data_loader, batch_total) as loader:
        # Set description to tqdm
        is_valid and loader.set_description("Validating")
        is_valid or loader.set_description("Testing   ")

        with torch.no_grad():
            # Batch Loop
            for idx, batch in enumerate(loader):
                # ---------------------- Main Process -----------------------
                data, target = (batch.text.to(device), batch.label.to("cpu"))
                target = target.detach().numpy().copy()

                outputs = model(data)
                outputs = torch.sigmoid(outputs)
                # -----------------------------------------------------------

                # Print some progress
                outputs = outputs.to("cpu").detach().numpy().copy()
                if "f1" in measure:
                    outputs = outputs >= 0.5

                target_all = np.concatenate([target_all, target])
                eval_all = np.concatenate([eval_all, outputs])

                if idx < batch_total - 1:
                    if "f1" in measure:
                        avg = measure[:-3]
                        eval_batch = f1_score(target, outputs, average=avg)
                        print_num_on_tqdm(loader, eval_batch, measure)
                    else:
                        # Calculate multiple precision values for this batch
                        p1_batch = precision_k(target, outputs, 1)
                        p3_batch = precision_k(target, outputs, 3)
                        p5_batch = precision_k(target, outputs, 5)
                        
                        # Show primary metric in progress bar
                        k = int(measure[-1])
                        if k == 1:
                            eval_batch = p1_batch
                        elif k == 3:
                            eval_batch = p3_batch
                        else:
                            eval_batch = p5_batch
                            
                        print_num_on_tqdm(loader, eval_batch, measure)
                else:
                    if "f1" in measure:
                        avg = measure[:-3]
                        eval_epoch = f1_score(target_all, eval_all, average=avg)
                        print_num_on_tqdm(loader, eval_epoch, measure, True)

                        # Log F1 metrics
                        if logger is not None:
                            metrics = {measure: eval_epoch}
                            if is_valid:
                                logger.log_validation_metrics(epoch, metrics)
                            else:
                                logger.log_test_metrics(metrics)
                    else:
                        # Calculate all precision metrics for the epoch
                        precision_values['p@1'] = precision_k(target_all, eval_all, 1)
                        precision_values['p@3'] = precision_k(target_all, eval_all, 3)
                        precision_values['p@5'] = precision_k(target_all, eval_all, 5)
                        
                        # Determine the primary metric for return value
                        k = int(measure[-1])
                        if k == 1:
                            eval_epoch = precision_values['p@1']
                        elif k == 3:
                            eval_epoch = precision_values['p@3']
                        else:
                            eval_epoch = precision_values['p@5']
                        
                        # Display all metrics
                        print_multiple_metrics(loader, precision_values, True)

                        # Log precision metrics
                        if logger is not None:
                            if is_valid:
                                logger.log_validation_metrics(epoch, precision_values)
                            else:
                                logger.log_test_metrics(precision_values)

    # Print the complete metric values at the end of validation/testing
    if not "f1" in measure:
        print(f"\nMetrics - P@1: {precision_values['p@1']:.6f}, P@3: {precision_values['p@3']:.6f}, P@5: {precision_values['p@5']:.6f}")

    return eval_epoch
