import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mask_fill_value):

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.1),
            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.1,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                rotate_limit=0,  # degrees
                scale_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=mask_fill_value,
                interpolation=cv2.INTER_CUBIC,
                p=0.1,
            ),
            A.ElasticTransform(
                alpha=60,
                sigma=8,
                value=0,
                mask_value=mask_fill_value,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_CUBIC,
                p=0.1,
            ),
            A.GridDistortion(
                num_steps=10,
                distort_limit=0.35,
                value=0,
                mask_value=mask_fill_value,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_CUBIC,
                p=0.1,
            ),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
            A.ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=0.01,  # must be in this interval [-0.5, 0.5].
                p=0.1,
            ),
            A.PixelDropout(p=0.1),
            A.CoarseDropout(p=0.1),
            A.ISONoise(p=0.1),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomGamma(p=0.1),
            A.ImageCompression(quality_lower=75, p=0.01),
            A.Perspective(fit_output=True, pad_mode=cv2.BORDER_CONSTANT, p=0.1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    valid_transforms = A.Compose([A.Normalize(), ToTensorV2()])

    test_transforms = A.Compose([A.Normalize(), ToTensorV2()])

    return train_transforms, valid_transforms, test_transforms
