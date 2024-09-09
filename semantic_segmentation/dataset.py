# Standard Library imports
import os

# External imports
import cv2
from torch.utils.data import Dataset


class SemanticSegmentationDataset(Dataset):
    """
    Generic Dataset class for semantic segmentation datasets.
    """

    def __init__(
        self,
        data_path,
        images_folder,
        masks_folder,
        image_ids,
        transforms=None,
    ):
        """
        Args:
            data_path (string): Path to the dataset folder.
            images_folder (string): Name of the folder containing the images.
            masks_folder (string): Name of the folder containing the masks.
            image_ids (list): List of image IDs to include in the dataset.
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """

        self.data_path = data_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_ids = image_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Get image and mask paths
        image_path = os.path.join(self.data_path, self.images_folder, f"{image_id}.jpg")
        mask_path = os.path.join(self.data_path, self.masks_folder, f"{image_id}.png")

        # Load image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            if mask is None:
                return self.transforms(image=image)["image"]
            else:
                transformed = self.transforms(image=image, mask=mask)

            return transformed["image"], transformed["mask"]

        return image, mask
