# Visual AI Microservice -- Project Progress Report

## 📌 Project Overview

A lightweight **Visual AI Microservice** built with **Python, PyTorch,
and FastAPI**.\
The service accepts an uploaded image and returns: - Predicted class
label(s)\
- Top-k probabilities\
- A blur/quality score\
- An optional segmentation mask (PNG) usable in GIMP/Inkscape\
- A REST endpoint for integrating into other products

The project mimics a real AI engineering workflow covering model
training, inference pipeline design, and API deployment.

## Completed Chapters / Features

### 1. Project Definition & Repository Structure

-   Input/output specification defined.
-   Repository structure initialized.

### 2. Environment & Tooling Setup (Linux Mint)

-   Python virtual environment created.
-   Installed core dependencies.

### 3. Data Collection & Preparation

-   Dataset downloaded and processed.
-   PyTorch DataLoader implemented.

### 4. Model Training (PyTorch)

-   ResNet18 pretrained backbone chosen.
-   Training and validation loops implemented.

### 5. Inference Pipeline

-   Unified analyze_image() function created.

### 6. FastAPI Service Implementation

-   POST /analyze-image endpoint implemented.
-   GET /health endpoint added.

### 7. Demo Client

-   CLI script added for testing API.

### 8. GIMP/Inkscape, Blender, Unity Integration

-   PNG masks loadable into external applications.

### 9. Documentation & Packaging

-   README includes installation and usage instructions.
