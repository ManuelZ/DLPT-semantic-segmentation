# Standard Library imports
import os
import gc
from collections import defaultdict

# External imports
import cv2
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch_lr_finder import LRFinder
from semantic_segmentation.optimizer import smart_optimizer


# Create colors for the visualization, one for each class
MASK_CLASS_COLORS = np.array(
    [
        [0, 0, 0],  # background
        [192, 128, 128],  # person
        [0, 128, 1],  # bike
        [128, 128, 128],  # car
        [128, 0, 0],  # drone
        [1, 0, 128],  # boat
        [193, 0, 129],  # animal
        [192, 0, 0],  # obstacle
        [192, 129, 0],  # construction
        [0, 65, 1],  # vegetation
        [127, 128, 0],  # road
        [0, 128, 129],  # sky
    ]
)

LABELS_NAMES_MAP = {
    0: "background",
    1: "person",
    2: "bike",
    3: "car",
    4: "drone",
    5: "boat",
    6: "animal",
    7: "obstacle",
    8: "construction",
    9: "vegetation",
    10: "road",
    11: "sky",
}


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()
        gc.collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        gc.collect()


def torch_to_cv2(image: torch.Tensor, is_mask=False) -> np.ndarray:
    """
    Convert a PyTorch image tensor to an OpenCV image.
    Do not use on masks. Masks have to go from 3 dim (including batch) to 2 dim (no batch dim)
    """
    if is_mask:
        if image.ndim == 3:
            image = image.squeeze()
        image = image.to(torch.uint8)

    else:
        if image.ndim == 4:
            image = image.squeeze()
        image = image.permute(1, 2, 0)

    return image.cpu().numpy()


def extract_and_onehot_encode_classes_from_multilabel_masks(dataset, num_classes):
    """
    Extract the classes available in each multilabel binary mask and one-hot encodes them.

    Returns:
        A NumPy array of one-hot encoded classes, where the dimensions are
        (num_images, num_classes). Each value in the array represents the presence
        (1) or absence (0) of a specific class in the corresponding image.
    """
    ys = []
    for i in tqdm(range(len(dataset))):
        image, mask = dataset[i]
        y = np.unique(mask).reshape(1, -1)
        y = torch.Tensor(y).to(torch.int64)
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = y.sum(axis=1)
        y = y.numpy()
        ys.append(y)

    return np.concatenate(ys, axis=0)


def denormalize(tensors):
    """
    Denormalize image tensors back to range [0.0, 1.0]

    Modified from: Deep Learning with PyTorch - OpenCV University
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    for c in range(3):
        if len(tensors.shape) == 4:
            tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
        elif len(tensors.shape) == 3:
            tensors[c, :, :].mul_(std[c]).add_(mean[c])
        else:
            raise Exception(
                "Can only deal with images of shape (N, C, H, W) or (C, H, W)"
            )

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


def find_best_lr(
    model,
    loss_fun,
    dataloader,
    grad_accum_steps,
    device,
    start_lr=1e-7,
    end_lr=1,
    num_iter=200,
    momentum=None,
):
    """ """
    temp_optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=momentum)

    lr_finder = LRFinder(model, temp_optimizer, loss_fun, device=device)
    lr_finder.range_test(
        dataloader,
        end_lr=end_lr,
        num_iter=num_iter,
        accumulation_steps=grad_accum_steps,
    )
    lr_finder.plot()
    lr_finder.reset()

    best_lr = extract_best_lr(lr_finder)
    return best_lr


def extract_best_lr(lr_finder):
    """
    Extract the best Learning Rate for a trained LRFinder object.
    """

    learning_rates = np.array(lr_finder.history["lr"])
    losses = np.array(lr_finder.history["loss"])

    min_grad_idx = None
    try:
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
    except ValueError:
        print("Failed to compute the gradients, there might not be enough points.")

    if min_grad_idx is not None:
        best_lr = learning_rates[min_grad_idx]

    return best_lr


def calculate_class_weights(pixel_count_per_class):
    """
    Prepare class weights for BCE loss based on pixel counts, giving higher importance to classes
    with fewer pixels.

        loss_fun = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).to(torch.float32)
        )

    """

    total_pixels = np.sum(pixel_count_per_class)
    pixel_proportion_per_class = pixel_count_per_class / total_pixels

    # Calculate the inverse of the pixel proportions
    class_weights = 1.0 / (
        pixel_proportion_per_class + 1e-12
    )  # Adding epsilon to avoid division by zero

    # Normalize the class weights to make them sum to 1
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


def count_pixels_per_class(images_ids, datapath, num_classes):
    """ """

    pixel_count_per_class = np.zeros(num_classes)

    for image_id in tqdm(images_ids):
        mask_path = os.path.join(datapath, "masks/masks", f"{image_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for i in range(num_classes):
            class_pixels = np.sum(mask == i)
            pixel_count_per_class[i] += class_pixels

    return pixel_count_per_class


def count_images_per_class(dataset):
    """ """

    d = defaultdict(int)
    for i in tqdm(range(len(dataset))):
        image, mask = dataset[i]
        classes = np.unique(mask)
        for c in classes:
            d[c] += 1

    return d


def prepare_for_prediction(image_path, transforms, device):
    """ """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)["image"]
    image = image.to(device, dtype=torch.float32)
    image = image.unsqueeze(0)

    return image


def get_prediction(model, image, device):
    """ """
    image = image.to(device, dtype=torch.float32)
    image = image.unsqueeze(0)
    pred = model(image)["out"]
    pred = pred.argmax(dim=1)
    return pred
