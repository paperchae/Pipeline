import os
import torch


def set_optimizer(model, config):
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    else:
        raise ValueError("Invalid optimizer")
    return optimizer


def set_scheduler(optimizer, config):
    if config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    else:
        raise ValueError("Invalid scheduler")
    return scheduler


def save_model(valid_costs: list, save_point: list, model, **kwargs) -> bool:
    """
    If the current model has the lowest validation cost, save the model and remove the prior model.
    *If current epoch is 0, does not save the model.

    :param valid_costs: validation cost list
    :param save_point: save point list (model save time)
    :param model: model to save
    :param kwargs: current epoch, model save path
    :return: True if the model is saved, False if not for plotting loss graph
    """
    if kwargs["epoch"] != 0:
        # if train_costs[-1] < min(train_costs[:-1]) and valid_costs[-1] < min(valid_costs[:-1]):
        if valid_costs[-1] < min(valid_costs[:-1]):
            save_path = kwargs["model_save_path"] + "cost_{}_time_{}.pt".format(
                valid_costs[-1], save_point[-1]
            )
            torch.save(model.state_dict(), save_path)
            print("\nsaved model: {}".format(save_path))
            if kwargs["epoch"] > 1:
                try:
                    prior_cost = min(valid_costs[:-1])
                    prior_path = kwargs[
                                     "model_save_path"
                                 ] + "cost_{}_time_{}.pt".format(prior_cost, save_point[-2])
                    os.remove(prior_path)
                    print("removed prior model: {}".format(prior_path))
                except:
                    print("failed to remove prior model")
            return True
    else:
        return False


def early_stopping(valid_costs: list, n: int = 10) -> bool:
    """
    Early stopping if the validation cost is not decreased for 10 epochs.

    :param valid_costs: validation cost list
    :param n: number of epochs to check
    :return: True if early stopping, False if not
    """
    if len(valid_costs) > n:
        if min(valid_costs[-n:]) > min(valid_costs[:-n]):
            return True
        else:
            return False
    else:
        return False
