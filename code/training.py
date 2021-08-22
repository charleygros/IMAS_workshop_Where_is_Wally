import torch.nn.functional as F


def train_model(model, optimizer, dataset_training, dataset_validation=None, n_epoch=10):
    """Trains model."""
    idx = 0
    for i in range(n_epoch):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_bb in dataset_training:
            batch = x.shape[0]
            x = x.cuda().float()
            y_bb = y_bb.cuda().float()
            out_bb = model(x)
            loss = F.l1_loss(out_bb, y_bb, reduction="none").sum(1).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        if dataset_validation is not None:
            val_loss, val_acc = run_validation(model, dataset_validation)
            print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        else:
            print("train_loss %.3f" % (train_loss))
    return sum_loss/total


def run_validation(model, dataset_validation):
    """Runs model on the validation dataset and compute validation metrics."""
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_bb in dataset_validation:
        batch = y_bb.shape[0]
        x = x.cuda().float()
        y_bb = y_bb.cuda().float()
        out_bb = model(x)
        loss = F.l1_loss(out_bb, y_bb, reduction="none").sum(1).sum()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total


def update_optimizer(optimizer, lr):
    """Changes learning rate."""
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
