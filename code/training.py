import torch.nn as nn


def train_model(model, optimizer, dataset_training, dataset_validation=None, n_epoch=10):
    """Trains model."""
    idx = 0
    best_val_loss = -1
    loss_fct = DiceLoss().cuda()
    for i in range(n_epoch):
        model.train()
        total = 0
        sum_loss = 0
        for x, y in dataset_training:
            batch = x.shape[0]
            x = x.cuda().float()
            y = y.cuda().float()
            pred = model(x)
            pred = pred[:, 1:, :, :]
            loss = loss_fct(pred, y)
            print(loss)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss / total
        if dataset_validation is not None:
            val_loss = run_validation(model, dataset_validation, loss_fct)
            print("Epoch#{} -- Train loss: {} -- Validation loss: {}".format(i, round(train_loss, 3), round(val_loss, 3)))
            if val_loss < best_val_loss:
                best_model = model
                best_val_loss = val_loss
        else:
            print("Epoch#{} -- Train loss: {}".format(i, round(train_loss, 3)))
            best_model = model
    return best_model


def run_validation(model, dataset_validation, loss_fct):
    """Runs model on the validation dataset and compute validation metrics."""
    model.eval()
    total = 0
    sum_loss = 0
    for x, y in dataset_validation:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda().float()
        pred = model(x)
        pred = pred[:, 1:, :, :]
        loss = loss_fct(pred, y)
        sum_loss += loss.item()
        total += batch
    return sum_loss/total


def update_optimizer(optimizer, lr):
    """Changes learning rate."""
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


class DiceLoss(nn.Module):
    """DiceLoss.
    Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for
        volumetric medical image segmentation." 2016 fourth international conference on 3D vision (3DV). IEEE, 2016.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return - (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
