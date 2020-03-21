import wandb
import datetime
import time
import torch


def training_loop(model, optimizer, scheduler, val_dl, train_dl, criterion, config, DEVICE, prefix, tz_NY):

    epoch_amount = []
    train_loss_db = []
    train_acc_db = []
    val_loss_db = []
    val_acc_db = []
    loaded_epochs = 0
    flag_count = 0
    acc_weight_db = {}
    N_EPOCHS = 100


    for epoch in range(N_EPOCHS):

        all_paths = []
        all_ground_truths = []
        all_predictions = []

        c0_total, c1_total, c2_total, c3_total, c4_total = (0,) * 5
        c0_acc, c1_acc, c2_acc, c3_acc, c4_acc = (0,) * 5
        c0, c1, c2, c3, c4 = (0,) * 5

        # Training

        # activating train mode
        # wandb.watch(model)
        model.train()

        total_loss, n_correct, n_samples = 0.0, 0, 0
        init_time = time.time()

        # X = batch of images, y = labels
        # iterate for every batch of the training dataloader.

        for image, label, path in train_dl:
            # send to gpu for computations
            # the model must be on the gpu as well
            image, label = image.to(DEVICE), label.to(DEVICE)

            # zero the gradients of the optimizer
            # to prevent the accumulation of gradients
            # at each step
            optimizer.zero_grad()

            # we pass our data X trough our model
            # and get a prediction y_
            pred_label = model(image)

            # we use the loss function to calculate the loss
            # the difference between the y and y_, the predicted y
            loss = criterion(pred_label, label)

            # backpropagation, gives us the gradients
            loss.backward()

            # we optimize all the parameters of the model
            # the optimizer is define prior.
            optimizer.step()
            scheduler.step()

            _, y_label_ = torch.max(pred_label, 1)
            n_correct += (y_label_ == label).sum().item()
            total_loss += loss.item() * image.shape[0]
            n_samples += image.shape[0]

            train_loss = total_loss / n_samples
            train_acc = n_correct / n_samples * 100

        train_loss_db.append(train_loss)
        train_acc_db.append(train_acc)

        # Eval-------------------------------------------------------------------------------------------

        # activating eval mode, turns off dropout
        # batch norm behaves differently
        model.eval()

        total_loss, n_correct, n_samples = 0.0, 0, 0

        # no_grad = we don't need to track the gradients
        with torch.no_grad():
            # init_time = time.time()

            # iterate for every batch of the eval dataloader.
            for image, label, path in val_dl:

                image, label = image.to(DEVICE), label.to(DEVICE)
                pred_label = model(image)
                loss = criterion(pred_label, label)

                # epoch stats
                _, y_label_ = torch.max(pred_label, 1)
                n_correct += (y_label_ == label).sum().item()
                total_loss += loss.item() * image.shape[0]
                n_samples += image.shape[0]

                # passing to cpu
                prediction_cpu = pred_label.cpu().argmax(dim=1, keepdim=True)

                # pulling ground truths from preds
                ground_truths = label.view_as(prediction_cpu)

                # append paths
                for p in path:
                    all_paths.append(p)

                # Confusion Matrix
                for gt in ground_truths:
                    all_ground_truths.append(gt.item())

                for pc in prediction_cpu:
                    all_predictions.append(pc.item())

            for a, b, c in zip(all_ground_truths, all_predictions, all_paths):
                real_class = int(c.split('/')[-2])

                if real_class == 0:
                    c0_total += 1
                    if a == b:
                        c0 += 1
                        c0_acc = round((c0 / c0_total) * 100, 2)

                elif real_class == 1:
                    c1_total += 1
                    if a == b:
                        c1 += 1
                        c1_acc = round((c1 / c1_total) * 100, 2)

                elif real_class == 2:
                    c2_total += 1
                    if a == b:
                        c2 += 1
                        c2_acc = round((c2 / c2_total) * 100, 2)

                elif real_class == 3:
                    c3_total += 1
                    if a == b:
                        c3 += 1
                        c3_acc = round((c3 / c3_total) * 100, 2)

                elif real_class == 4:
                    c4_total += 1
                    if a == b:
                        c4 += 1
                        c4_acc = round((c4 / c4_total) * 100, 2)

                # generating status variables
                epoch_time = time.time() - init_time
                epoch_amount.append(epoch + loaded_epochs + 1)

                val_loss = total_loss / n_samples
                # val_acc = n_correct / n_samples * 100

        val_acc = round(((c0_acc + c1_acc + c2_acc + c3_acc + c4_acc) / 5), 2)
        val_acc_db.append(val_acc)
        val_loss_db.append(val_loss)

        # get current lr
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']

        # if epoch_amount[-1] > 1:
        #   if val_acc_db[-1] < (val_acc_db[-2]):
        #     flag_count+=1

        #   if val_acc_db[-1] == (val_acc_db[-2]):
        #     flag_count+=1

        # backup if best acc
        if len(val_acc_db) > 5 and val_acc_db[-1] > max(val_acc_db[:-1]):
            results_prefix = f'{prefix}_{epoch_amount[-1]}_{round(val_acc_db[-1])}%'
            filename_of_state_dict = results_prefix + '_state_dict.pt'
            torch.save(model.state_dict(), filename_of_state_dict)

            #filename_of_model = results_prefix + '_trained_model.pt'
            #acc_weight_db[val_acc_db[-1]] = './' + filename_of_state_dict

            datetime_NY = datetime.now(tz_NY)

        print(
                f"{epoch_amount[-1]}/{N_EPOCHS} | "
                f"{datetime_NY.strftime('%H:%M:%S')} | "
                f"valid acc: {val_acc:9.3f}% | "
                f"current lr: {old_lr:9.8f} | "
                f"flag count: {flag_count} | "
                f"{(epoch_time // 60):9.1f}min")

        print(
                f"{epoch_amount[-1]}/{N_EPOCHS} | "
                f"{datetime_NY.strftime('%H:%M:%S')} | "
                f'class 0: {c0_acc}% | '
                f'class 1: {c1_acc}% | '
                f'class 2: {c2_acc}% | '
                f'class 3: {c3_acc}% | '
                f'class 4: {c4_acc}%\n')

        # custom scheduler
        # if flag_count == 5:
        #   for param_group in optimizer.param_groups:
        #     new_lr = old_lr * 0.50
        #     param_group['lr'] = new_lr
        #     flag_count = 0
        #     model.load_state_dict(torch.load(acc_weight_db[max(acc_weight_db)]))ooin_loss_db[-1]})

        wandb.log({"Val Accuracy": val_acc, "Val Loss": val_loss_db[-1],
                   "Train Accuracy": train_acc_db[-1], "Learning Rate": old_lr})