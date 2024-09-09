# Standard Library imports
import os

# External imports
import torch
import numpy as np
from tqdm.autonotebook import tqdm


def main(
    model,
    optimizer,
    scheduler,
    loss_fun,
    scorer,
    train_dataloader,
    valid_dataloader,
    epochs,
    output_path,
    grad_accum_steps,
    batch_size,
    device,
):

    H = {
        "train_loss": [],
        "train_score": [],
        "valid_loss": [],
        "valid_score": [],
        "per_class_score": [],
    }
    best_score = 0

    for e in range(0, epochs):

        print("\n[INFO] EPOCH: {}/{}".format(e + 1, epochs))

        avg_train_loss, avg_train_score = train(
            model,
            optimizer,
            scheduler,
            loss_fun,
            scorer,
            train_dataloader,
            grad_accum_steps,
            batch_size,
            device,
        )

        avg_valid_loss, avg_valid_score, avg_per_class_score = test(
            model, loss_fun, scorer, valid_dataloader, batch_size, device
        )

        H["train_loss"].append(avg_train_loss)
        H["valid_loss"].append(avg_valid_loss)
        H["train_score"].append(avg_train_score)
        H["valid_score"].append(avg_valid_score)
        H["per_class_score"].append(avg_per_class_score)

        print(
            "Epoch train loss: {:.6f} | Epoch train mean Dice score: {:.4f}".format(
                avg_train_loss, avg_train_score
            )
        )
        print(
            "Epoch valid loss: {:.6f} | Epoch valid mean Dice score: {:.4f}".format(
                avg_valid_loss, avg_valid_score
            )
        )

        if avg_valid_score > best_score:
            best_score = avg_valid_score
            print(f"New best valid mean Dice score: {best_score:.4f} at epoch {e+1}")
            output_file_path = os.path.join(output_path, f"deeplabv3_best_model.pkl")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_file_path,
            )

        # Serialize the model every 5 epochs
        if (e + 1) % 5 == 0:
            output_file_path = os.path.join(
                output_path, f"deeplabv3_model_epoch_{e+1}.pkl"
            )
            torch.save(model, output_file_path)

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
):
    """ """

    model.train()

    train_steps = len(dataloader.dataset) // batch_size
    train_prog_bar = tqdm(dataloader, total=train_steps)
    epoch_loss = 0
    epoch_score = 0
    for batch_index, (x, y) in enumerate(train_prog_bar):

        x = x.to(device, dtype=torch.float32)
        y = y.squeeze()
        y = y.to(device, dtype=torch.long)

        pred_logits = model(x)["out"]

        # Train loss
        train_loss = loss_fun(pred_logits, y)
        # For en explanation of this, see "MLOps Engineering at Scale-Manning (2022), Ch 8.1.3"
        train_loss = train_loss / grad_accum_steps
        epoch_loss += train_loss.item()
        train_loss.backward()

        # Train score
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

        train_prog_bar.set_description(
            desc=f"Training loss: {train_loss.item():.4f} | Mean Dice score: {float(train_score.mean()):.2f}"
        )

    # Average train metrics during the epoch
    avg_train_loss = epoch_loss / train_steps
    avg_train_score = epoch_score / train_steps

    return avg_train_loss, avg_train_score


def test(model, loss_fun, scorer, dataloader, batch_size, device):

    epoch_loss = 0
    epoch_score = 0
    per_class_score = []

    model.eval()
    with torch.no_grad():

        valid_steps = len(dataloader.dataset) // batch_size
        valid_prog_bar = tqdm(dataloader, total=valid_steps)
        for x, y in valid_prog_bar:
            x = x.to(device, dtype=torch.float32)
            y = y.squeeze()
            y = y.to(device, dtype=torch.long)

            # Validation loss
            pred_logits = model(x)["out"]
            valid_loss = loss_fun(pred_logits, y)
            epoch_loss += valid_loss.item()

            # Validation score
            pred_probs = pred_logits.softmax(dim=1)
            max_indices = pred_probs.argmax(dim=1)
            valid_score = scorer(max_indices, y)
            epoch_score += float(valid_score.mean())

            # Per-class validation score
            per_class_score.append(valid_score.reshape(1, -1))

            valid_prog_bar.set_description(
                desc=f"Validation loss: {valid_loss.item():.4f} | Mean Dice score: {float(valid_score.mean()):.2f}"
            )

        # Average validation metrics during the epoch
        avg_valid_loss = epoch_loss / valid_steps
        avg_valid_score = epoch_score / valid_steps
        avg_per_class_score = (
            np.concatenate(per_class_score, axis=0).sum(axis=0) / valid_steps
        )
    return avg_valid_loss, avg_valid_score, avg_per_class_score
