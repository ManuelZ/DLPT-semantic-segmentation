# Standard Library imports
import os
from pathlib import Path

# External imports
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(
    model,
    optimizer,
    scheduler,
    loss_fun,
    scorer,
    train_dataloader,
    valid_dataloader,
    starting_epoch,
    epochs,
    output_path,
    grad_accum_steps,
    batch_size,
    device,
    use_aux=False,
):

    H = {
        "train_loss": [],
        "train_score": [],
        "valid_loss": [],
        "valid_score": [],
        "per_class_score": [],
    }
    best_score = 0

    writer = SummaryWriter()

    for e in range(starting_epoch, epochs):

        print("\n[INFO] EPOCH: {}/{}".format(e + 1, epochs))

        train_loss, train_score = train(
            model,
            optimizer,
            scheduler,
            loss_fun,
            scorer,
            train_dataloader,
            grad_accum_steps,
            batch_size,
            device,
            use_aux,
        )

        valid_loss, valid_score, avg_per_class_score = test(
            model, loss_fun, scorer, valid_dataloader, batch_size, device
        )

        writer.add_scalar("Loss/train", train_loss, e)
        writer.add_scalar("Loss/val", valid_loss, e)

        writer.add_scalar("Score/train", train_score, e)
        writer.add_scalar("Score/val", valid_score, e)

        H["train_loss"].append(train_loss)
        H["valid_loss"].append(valid_loss)
        H["train_score"].append(train_score)
        H["valid_score"].append(valid_score)
        H["per_class_score"].append(avg_per_class_score)

        print(
            "Epoch train loss: {:.6f} | Epoch train mean Dice score: {:.4f}".format(
                train_loss, train_score
            )
        )
        print(
            "Epoch valid loss: {:.6f} | Epoch valid mean Dice score: {:.4f}".format(
                valid_loss, valid_score
            )
        )

        if valid_score > best_score:
            best_score = valid_score
            print(f"New best valid mean Dice score: {best_score:.4f} at epoch {e+1}")

            if not Path(output_path).exists():
                Path(output_path).mkdir(parents=True, exist_ok=True)

            output_file_path = os.path.join(output_path, f"deeplabv3_best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_file_path,
            )

    return H


def train(
    model,
    optimizer,
    scheduler,
    loss_fun,
    scorer,
    dataloader,
    grad_accum_steps,
    batch_size,
    device,
    use_aux=False,
):
    """ """

    model.train()

    epoch_loss = 0
    epoch_score = 0

    num_steps = len(dataloader.dataset) // batch_size
    prog_bar = tqdm(dataloader, total=num_steps)
    for batch_index, (x, y) in enumerate(prog_bar):

        x = x.to(device, dtype=torch.float32)
        y = y.squeeze()
        y = y.to(device, dtype=torch.long)

        output = model(x)

        pred_logits = output["out"]
        if use_aux:
            aux_loss = output["aux"]

        loss = loss_fun(pred_logits, y)

        # For en explanation of this, see "MLOps Engineering at Scale-Manning (2022), Ch 8.1.3"
        loss /= grad_accum_steps
        if use_aux:
            aux_loss /= grad_accum_steps
            loss += aux_loss

        epoch_loss += loss.item()
        loss.backward()

        pred_probs = pred_logits.softmax(dim=1)
        max_indices = pred_probs.argmax(dim=1)
        train_score = scorer(max_indices, y)
        epoch_score += float(train_score.mean())

        # Gradient accumulation
        if ((batch_index + 1) % grad_accum_steps == 0) or (
            batch_index + 1 == len(dataloader)
        ):

            # Weights update
            optimizer.step()

            # Optimizer Learning Rate update
            scheduler.step()

            optimizer.zero_grad()  # TODO: test passing set_to_none=True

        prog_bar.set_description(
            desc=f"Training loss: {loss.item():.4f} | score: {float(train_score.mean()):.2f}"
        )

    # Average train metrics during the epoch
    avg_loss = epoch_loss / num_steps
    avg_score = epoch_score / num_steps

    return avg_loss, avg_score


def validate(model, loss_fun, scorer, dataloader, batch_size, device):
    """ """

    epoch_loss = 0
    epoch_score = 0
    per_class_score = []

    model.eval()
    with torch.no_grad():

        num_steps = len(dataloader.dataset) // batch_size
        prog_bar = tqdm(dataloader, total=num_steps)
        for x, y in prog_bar:
            x = x.to(device, dtype=torch.float32)
            y = y.squeeze()
            y = y.to(device, dtype=torch.long)

            # Loss
            pred_logits = model(x)["out"]
            loss = loss_fun(pred_logits, y)
            epoch_loss += loss.item()

            # Score
            pred_probs = pred_logits.softmax(dim=1)
            max_indices = pred_probs.argmax(dim=1)
            score = scorer(max_indices, y)
            epoch_score += float(score.mean())

            # Per-class validation score
            per_class_score.append(score.reshape(1, -1))

            prog_bar.set_description(
                desc=f"Validation loss: {loss.item():.4f} | score: {float(score.mean()):.2f}"
            )

        # Average validation metrics during the epoch
        avg_loss = epoch_loss / num_steps
        avg_score = epoch_score / num_steps
        avg_per_class_score = (
            np.concatenate(per_class_score, axis=0).sum(axis=0) / num_steps
        )
    return avg_loss, avg_score, avg_per_class_score
