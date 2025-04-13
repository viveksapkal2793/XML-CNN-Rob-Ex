import numpy as np
import torch
from scipy import stats as stats
from sklearn.metrics import f1_score
from torch import nn as nn

from my_functions import precision_k, print_num_on_tqdm, tqdm_with_num, print_multiple_metrics
from adversarial_defense import FGSM
import warnings
warnings.filterwarnings("ignore")

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

def adversarial_training(params, model, train_loader, optimizer, epoch=1, logger=None):
    device = params["device"]
    batch_total = params["train_batch_total"]
    loss_func = nn.BCELoss()
    
    # Initialize FGSM attack
    fgsm = FGSM(model, epsilon=params.get("fgsm_epsilon", 0.1))

    model.train()
    clean_losses = []
    adv_losses = []

    # Show loss with tqdm
    with tqdm_with_num(train_loader, batch_total) as loader:
        loader.set_description("Adv Train ")

        # Batch Loop
        for idx, batch in enumerate(loader):
            # ---------------------- Clean Example Training -----------------------
            data, target = (batch.text.to(device), batch.label.to(device))
            
            # Forward pass with clean examples
            optimizer.zero_grad()
            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            clean_loss = loss_func(outputs, target)
            
            # Backward pass to get gradients
            # clean_loss.backward()
            
            # Store clean loss
            clean_losses.append(clean_loss.item())
            
            # ---------------------- Adversarial Example Training -----------------------
            # Generate adversarial examples using FGSM
            perturbed_embeddings = fgsm.generate(data, target, loss_func)
            
            # Forward pass with adversarial examples
            # optimizer.zero_grad()
            
            # We need to manually run the forward pass since we're using embeddings directly
            h_non_static = perturbed_embeddings.unsqueeze(1)  # Add channel dimension
            h_non_static = model.dropout_0(h_non_static)
            
            # Process through model layers
            h_list = []
            for i in range(len(model.filter_sizes)):
                h_n = model.conv_layers[i](h_non_static)
                h_n = h_n.view(h_n.shape[0], 1, h_n.shape[1] * h_n.shape[2])
                h_n = model.pool_layers[i](h_n)
                h_n = torch.relu(h_n)
                h_n = h_n.view(h_n.shape[0], -1)
                h_list.append(h_n)
                
            if len(model.filter_sizes) > 1:
                h = torch.cat(h_list, 1)
            else:
                h = h_list[0]
                
            h = torch.relu(model.l1(h))
            h = model.dropout_1(h)
            adv_outputs = model.l2(h)
            adv_outputs = torch.sigmoid(adv_outputs)
            
            # Calculate loss on adversarial examples
            adv_loss = loss_func(adv_outputs, target)
            
            # Backward pass and optimize
            # adv_loss.backward()
            # optimizer.step()
            
            # Store adversarial loss
            adv_losses.append(adv_loss.item())
            
            # Total loss for logging
            # total_loss = clean_loss.item() + adv_loss.item()

            # Weighted total loss for training
            total_loss = 0.7*clean_loss + 0.3*adv_loss
            total_loss.backward()
            optimizer.step()
            
            # Log batch losses if logger is available
            if logger is not None:
                logger.log_train_loss(epoch, idx, total_loss.item(), 
                                     {"clean_loss": clean_loss.item(), "adv_loss": adv_loss.item()})

            if idx < batch_total - 1:
                print_num_on_tqdm(loader, total_loss)
            else:
                clean_loss_epoch = np.mean(clean_losses)
                adv_loss_epoch = np.mean(adv_losses)
                total_loss_epoch = clean_loss_epoch + adv_loss_epoch
                
                print(f"\nClean loss: {clean_loss_epoch:.6f}, Adv loss: {adv_loss_epoch:.6f}")
                print_num_on_tqdm(loader, total_loss_epoch, last=True)

                # Log average epoch losses
                if logger is not None:
                    logger.log_train_loss(epoch, avg_loss=total_loss_epoch,
                                        extra_metrics={"clean_loss": clean_loss_epoch, "adv_loss": adv_loss_epoch})

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



def explainability_data_extract(params, model, data_loader):
    # print(params)
    device = params["device"]
    model = model.to("cpu")
    model.eval()

    all_ids = []
    all_probs = []

    batch_total = params["test_batch_total"]

    with tqdm_with_num(data_loader, batch_total) as loader:
        loader.set_description("Explainiability data extraction")

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                inputs = batch.text.to("cpu")
                ids = batch.id  

                outputs = model(inputs)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()

                all_ids.extend(ids)
                all_probs.extend(probs)

    return list(zip(all_ids, all_probs)) 

def adversarial_validating_testing(params, model, data_loader, epoch=1, is_valid=True, logger=None):
    device = params["device"]
    measure = params["measure"]
    doc_key = is_valid and "valid" or "test"
    batch_total = params[doc_key + "_batch_total"]
    
    # Initialize FGSM attack
    fgsm = FGSM(model, epsilon=params.get("fgsm_epsilon", 0.1))
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()

    # Metrics for clean examples
    clean_eval_epoch = 0.0
    clean_target_all = np.empty((0, params["num_of_class"]), dtype=np.int8)
    clean_eval_all = np.empty((0, params["num_of_class"]), dtype=np.float32)
    clean_precision_values = {'p@1': 0.0, 'p@3': 0.0, 'p@5': 0.0}
    
    # Metrics for adversarial examples
    adv_eval_epoch = 0.0
    adv_target_all = np.empty((0, params["num_of_class"]), dtype=np.int8)
    adv_eval_all = np.empty((0, params["num_of_class"]), dtype=np.float32)
    adv_precision_values = {'p@1': 0.0, 'p@3': 0.0, 'p@5': 0.0}

    # First evaluate on clean examples
    print("\nEvaluating on clean examples:")
    with tqdm_with_num(data_loader, batch_total) as loader:
        # Set description to tqdm
        is_valid and loader.set_description("Clean Val ")
        is_valid or loader.set_description("Clean Test")

        with torch.no_grad():
            # Batch Loop
            for idx, batch in enumerate(loader):
                data, target = (batch.text.to(device), batch.label.to(device))
                # target_np = target.detach().numpy().copy()
                target_np = target.detach().cpu().numpy().copy()

                outputs = model(data)
                outputs = torch.sigmoid(outputs)
                # outputs_np = outputs.to("cpu").detach().numpy().copy()
                outputs_np = outputs.detach().cpu().numpy().copy()
                
                if "f1" in measure:
                    outputs_np = outputs_np >= 0.5

                clean_target_all = np.concatenate([clean_target_all, target_np])
                clean_eval_all = np.concatenate([clean_eval_all, outputs_np])

                if idx < batch_total - 1:
                    if "f1" in measure:
                        avg = measure[:-3]
                        eval_batch = f1_score(target_np, outputs_np, average=avg)
                        print_num_on_tqdm(loader, eval_batch, measure)
                    else:
                        k = int(measure[-1])
                        eval_batch = precision_k(target_np, outputs_np, k)
                        print_num_on_tqdm(loader, eval_batch, measure)
                else:
                    if "f1" in measure:
                        avg = measure[:-3]
                        clean_eval_epoch = f1_score(clean_target_all, clean_eval_all, average=avg)
                        print_num_on_tqdm(loader, clean_eval_epoch, measure, True)
                    else:
                        clean_precision_values['p@1'] = precision_k(clean_target_all, clean_eval_all, 1)
                        clean_precision_values['p@3'] = precision_k(clean_target_all, clean_eval_all, 3)
                        clean_precision_values['p@5'] = precision_k(clean_target_all, clean_eval_all, 5)
                        
                        k = int(measure[-1])
                        if k == 1:
                            clean_eval_epoch = clean_precision_values['p@1']
                        elif k == 3:
                            clean_eval_epoch = clean_precision_values['p@3']
                        else:
                            clean_eval_epoch = clean_precision_values['p@5']
                        
                        print_multiple_metrics(loader, clean_precision_values, True)

    # Then evaluate on adversarial examples
    print("\nEvaluating on adversarial examples:")
    with tqdm_with_num(data_loader, batch_total) as loader:
        # Set description to tqdm
        is_valid and loader.set_description("Adv Val   ")
        is_valid or loader.set_description("Adv Test  ")

        # Batch Loop
        for idx, batch in enumerate(loader):
            data, target = (batch.text.to(device), batch.label.to(device))
            target_np = target.detach().cpu().numpy().copy()
            
            # Generate adversarial examples
            loss_fn = nn.BCELoss()
            
            # We need to set requires_grad on inputs for FGSM
            with torch.enable_grad():
                perturbed_embeddings = fgsm.generate(data, target, loss_fn)
            
            # Forward pass with adversarial examples
            with torch.no_grad():
                # Process perturbed embeddings through the model
                h_non_static = perturbed_embeddings.unsqueeze(1)
                
                h_list = []
                for i in range(len(model.filter_sizes)):
                    h_n = model.conv_layers[i](h_non_static)
                    h_n = h_n.view(h_n.shape[0], 1, h_n.shape[1] * h_n.shape[2])
                    h_n = model.pool_layers[i](h_n)
                    h_n = torch.relu(h_n)
                    h_n = h_n.view(h_n.shape[0], -1)
                    h_list.append(h_n)
                    
                if len(model.filter_sizes) > 1:
                    h = torch.cat(h_list, 1)
                else:
                    h = h_list[0]
                    
                h = torch.relu(model.l1(h))
                outputs = model.l2(h)
                outputs = torch.sigmoid(outputs)
                
                # Apply feature squeezing if enabled
                if hasattr(model, 'feature_squeezing'):
                    outputs = model.feature_squeezing.squeeze(outputs)
                
                # outputs_np = outputs.to("cpu").detach().numpy().copy()
                outputs_np = outputs.detach().cpu().numpy().copy()
            
            if "f1" in measure:
                outputs_np = outputs_np >= 0.5

            adv_target_all = np.concatenate([adv_target_all, target_np])
            adv_eval_all = np.concatenate([adv_eval_all, outputs_np])

            if idx < batch_total - 1:
                if "f1" in measure:
                    avg = measure[:-3]
                    eval_batch = f1_score(target_np, outputs_np, average=avg)
                    print_num_on_tqdm(loader, eval_batch, measure)
                else:
                    k = int(measure[-1])
                    eval_batch = precision_k(target_np, outputs_np, k)
                    print_num_on_tqdm(loader, eval_batch, measure)
            else:
                if "f1" in measure:
                    avg = measure[:-3]
                    adv_eval_epoch = f1_score(adv_target_all, adv_eval_all, average=avg)
                    print_num_on_tqdm(loader, adv_eval_epoch, measure, True)
                else:
                    adv_precision_values['p@1'] = precision_k(adv_target_all, adv_eval_all, 1)
                    adv_precision_values['p@3'] = precision_k(adv_target_all, adv_eval_all, 3)
                    adv_precision_values['p@5'] = precision_k(adv_target_all, adv_eval_all, 5)
                    
                    k = int(measure[-1])
                    if k == 1:
                        adv_eval_epoch = adv_precision_values['p@1']
                    elif k == 3:
                        adv_eval_epoch = adv_precision_values['p@3']
                    else:
                        adv_eval_epoch = adv_precision_values['p@5']
                    
                    print_multiple_metrics(loader, adv_precision_values, True)

    # Log metrics
    if logger is not None:
        combined_metrics = {}
        
        if "f1" in measure:
            combined_metrics = {
                "clean_"+measure: clean_eval_epoch,
                "adv_"+measure: adv_eval_epoch
            }
        else:
            for k in ['p@1', 'p@3', 'p@5']:
                combined_metrics["clean_"+k] = clean_precision_values[k]
                combined_metrics["adv_"+k] = adv_precision_values[k]
        
        if is_valid:
            logger.log_validation_metrics(epoch, combined_metrics)
        else:
            logger.log_test_metrics(combined_metrics)
    
    # Print summary comparison
    if "f1" in measure:
        print(f"\nSummary - Clean {measure}: {clean_eval_epoch:.6f}, Adversarial {measure}: {adv_eval_epoch:.6f}")
        print(f"Robustness gap: {clean_eval_epoch - adv_eval_epoch:.6f}")
    else:
        print(f"\nSummary - Clean metrics:")
        print(f"P@1: {clean_precision_values['p@1']:.6f}, P@3: {clean_precision_values['p@3']:.6f}, P@5: {clean_precision_values['p@5']:.6f}")
        print(f"Adversarial metrics:")
        print(f"P@1: {adv_precision_values['p@1']:.6f}, P@3: {adv_precision_values['p@3']:.6f}, P@5: {adv_precision_values['p@5']:.6f}")
        
        k = int(measure[-1])
        if k == 1:
            gap = clean_precision_values['p@1'] - adv_precision_values['p@1']
        elif k == 3:
            gap = clean_precision_values['p@3'] - adv_precision_values['p@3']
        else:
            gap = clean_precision_values['p@5'] - adv_precision_values['p@5']
        print(f"Robustness gap ({measure}): {gap:.6f}")

    # Return the clean evaluation metric as the primary metric for model selection
    return clean_eval_epoch