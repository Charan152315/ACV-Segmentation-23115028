# Advanced Computer Vision(CS107203CS) — Assignment B
# Semantic Segmentation Project

**Name:** Charansri Chintamaneni  
**Roll No:** 23115028  
**Submitted To:** Dr. Bunil Kumar Balabantaray  

## Project Overview
This repository contains the code and logs for a reproducible semantic segmentation experiment on the **Kvasir-SEG** (gastrointestinal polyp) dataset. The goal of the experiment is to train a lightweight U-Net baseline and test the exact effect of changing only one mechanism—replacing standard bottleneck convolutions (`dilation = 1, 1`) with dilated convolutions (`dilation = 2, 4`)—while keeping the parameter count (`482,754`), random seed (`42`), and training setup completely fixed.

## Files in this Repository
* `train_and_eval.py` — Complete script that downloads Kvasir-SEG, trains both models, runs the evaluation, and generates the output plots.
* `training_log.csv` — Epoch-by-epoch training loss, validation loss, and validation IoU scores for both runs.
* `metrics_summary.json` — Final test set IoU, mIoU, parameter counts, Tesla T4 inference latency, and package versions.
* `qualitative_results.png` — Side-by-side visual predictions comparing the input frame, human ground truth, baseline model, and controlled model.
* `requirements.txt` — Python libraries used for the experiment.

## Key Test Results (Tesla T4 GPU)
* **Baseline U-Net (`dilation = 1, 1`):** `59.39%` mIoU (`33.96%` Polyp IoU, `84.81%` Background IoU, `2.84 ms/img` latency)
* **Dilated Bottleneck U-Net (`dilation = 2, 4`):** `63.64%` mIoU (`43.12%` Polyp IoU, `84.15%` Background IoU, `2.96 ms/img` latency)

## How to Run
```bash
pip install -r requirements.txt
python train_and_eval.py

