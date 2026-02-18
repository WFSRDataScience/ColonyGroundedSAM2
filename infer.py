import os
import torch

import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
import argparse
from utils import load_image2
import supervision as sv
import matplotlib.pyplot as plt


def load_models():
    SAM2_CHECKPOINT = "Grounded-SAM-2-main/checkpoints/sam2_hiera_large.pt"
    SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "Grounded-SAM-2-main/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "checkpoints/colony_gd.pth"

    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device='cuda'
    )

    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return grounding_model, sam2_predictor

def infer_colony_grounded_sam2(data_path, out_path, crop_coords=None, box_threshold=0.25, text_threshold=0.25):
    grounding_model, sam2_predictor = load_models()
    data_paths = os.listdir(data_path)
    for i, img_path in enumerate(data_paths):

        # Decide which coordinates to use for cropping the image
        if len(crop_coords) == len(data_paths):
            x1, x2, y1, y2 = crop_coords[i]
        elif len(crop_coords) == 4:
            x1, x2, y1, y2 = crop_coords
        else:
            image_shape = cv2.imread(os.path.join(data_path, img_path)).shape
            x1, x2, y1, y2 = (0, image_shape[1], 0, image_shape[0])

        # Load image and crop if necessary
        image_source, image = load_image2(os.path.join(data_path, img_path), x1, x2, y1, y2)

        # Forward pass through finetuned grounding dino
        boxes, confidences, labels = predict(
        model=grounding_model,
        image= image,
        caption="microbial colony.",
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
        
        # Convert found boxes to correct format
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Calculate image embeddings
        sam2_predictor.set_image(image_source)

        # For memory constraints, we forward pass the objects in batches, increase batch_size for speed
        batch_size = 100
        all_masks = []

        for i in range(0, len(input_boxes), batch_size):
            batch_boxes = input_boxes[i:i+batch_size]

            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=batch_boxes,
                multimask_output=False,
            )
            if len(masks.shape) == 3:
                masks = masks.reshape(1, masks.shape[0], masks.shape[1], masks.shape[2])
            all_masks.append(masks)

        masks = np.concatenate(all_masks, axis=0)

        # Show annotated image
        img = cv2.imread(os.path.join(data_path, img_path))[:,:,::-1]
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id = np.array(list(range(len(labels))))
        )

        plt.figure(figsize=(15,15))
        plt.axis('off')
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

        # We save each annotated image as a visual inspection, for the raw segmentations please use 'masks'
        plt.imsave(os.path.join(out_path, 'annotated_'+img_path), annotated_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--out_path')
    parser.add_argument('--crop_coords', default=None)
    parser.add_argument('--box_threshold', default=0.25)
    parser.add_argument('--text_threshold', default=0.25)

    args = parser.parse_args()
    infer_colony_grounded_sam2(args.data_path, args.out_path, args.crop_coords, args.box_threshold, args.text_threshold)