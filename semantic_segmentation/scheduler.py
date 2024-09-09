from torch.optim import lr_scheduler


def get_scheduler(scheduler_name, optimizer, total_steps, max_lr=None, min_lr=None):
    """ """

    if scheduler_name == "constant":
        return lr_scheduler.LinearLR(optimizer, start_factor=1, total_iters=total_steps)

    elif scheduler_name == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )

    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps / 10, eta_min=min_lr
        )
