# Colony Grounded SAM2

This is the official repository for the paper Colony Grounded SAM2: Zero-shot detection and segmentation of bacterial colonies using foundation models, from the SPIE medical imaging 2026 conference.


<img width="10517" height="4242" alt="workflow_spie_korp25" src="https://github.com/user-attachments/assets/5db841dc-68fe-4d31-9dd0-66370dfb5de0" />

If you find this useful and use our code or database, please cite our paper:

> Korporaal, Daan, et al. "Colony Grounded SAM2: Zero-shot detection and segmentation of bacterial colonies using foundation models." *SPIE Medical Imaging* (2026).


Or use the bibtex: 

```
@inproceedings{korporaal2026colony,
  title={Colony Grounded SAM2: Zero-shot detection and segmentation of bacterial colonies using foundation models},
  author={Korporaal, D and de Kruijf, P and Litjens, Ralph and van der Velden, BHM},
  booktitle={SPIE Medical Imaging 2026},
  year={2026}
}
```


## Installation
As this work highly builds upon IDEA's Grounded SAM2, we first download their repository: 
### 1. Clone Grounded SAM2

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

---

### 2. Install Dependencies

The repository recommends Python 3.10 with PyTorch and TorchVision installed. 
Either use the docker image as provided by Grounded SAM2 or run:


```bash
pip install torch torchvision torchaudio
```

Install Segment Anything 2:

```bash
pip install -e .
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e grounding_dino
```

If you plan to run GPU inference, ensure your CUDA environment variables are set correctly for cuda>=12.1.

---

### 3. Download SAM2 checkpoint


```bash
cd checkpoints
bash download_ckpts.sh
```
### 4. Download finetuned Grounding DINO weights
Return to the Colony Grounded SAM2 workspace and run:

```bash
wget https://huggingface.co/DataScienceWFSR/ColonyGroundedSAM2/resolve/main/checkpoints/colony_gd.pth -P checkpoints
```

## Running Inference

Use `infer.py` to run the fine-tuned Colony Grounded SAM2 model. The script accepts the following arguments:

```bash
python infer.py \
    --data_path <PATH_TO_INPUT_IMAGES_FOLDER> \
    --out_path <PATH_TO_SAVE_OUTPUTS_FOLDER> \
    [--crop_coords x1 y1 x2 y2] \
    [--box_threshold 0.25] \
    [--text_threshold 0.25]
```

### Arguments

- `--data_path` : Path to input image or folder containing images  
- `--out_path` : Path to save outputs (segmentation masks, boxes, etc.)  
- `--crop_coords` : Optional crop coordinates (`x1 y1 x2 y2`) to focus on a region, either a single set of coordinates, a list of size num_images with sets of coordinates, or None for no cropping
- `--box_threshold` : Detection box confidence threshold (default: 0.25)  
- `--text_threshold` : Text grounding confidence threshold (default: 0.25)  


