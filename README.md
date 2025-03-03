# Dynamic Story Visualizer

## Overview

This project integrates multiple AI models into a creative pipeline that transforms story text into stylized images. It combines natural language processing, deep learning, and computer vision to generate visually compelling storytelling experiences.

## Pipeline Structure

### 1. Text Analysis Pipeline

- **Input**: Raw text from the user's story
- **Process**:
  - Sentence splitting (NLTK)
  - Emotion analysis (DistilRoBERTa)
  - Semantic embedding (SentenceTransformer)
  - Keyword matching
- **Output**: Style selection for each sentence with analysis details

### 2. Image Generation Pipeline

- **Input**: Individual sentences with selected styles
- **Process**:
  - Text-to-image generation (Stable Diffusion)
  - Style transfer from art masterpieces (TensorFlow model)
- **Output**: Pairs of content images and stylized images

### 3. User Interface Pipeline

- **Input**: Processed images and analysis data
- **Process**: Display in Gradio UI with appropriate galleries and text output
- **Output**: Visual story with debug information

## Key Features

This application enables:

1. **Understanding the emotional and semantic content** of each sentence in a story through advanced NLP techniques.
2. **Matching appropriate artistic styles** to each scene based on its mood and content.
3. **Generating realistic images** that depict the scene's content.
4. **Applying artistic styling** inspired by famous artists like Van Gogh, Munch, and Monet.
5. **Providing analysis transparency** through detailed debugging information.

## Technologies Used

This project integrates multiple AI domains:

- **Natural Language Processing (NLP)**: Sentence parsing, emotion detection, and semantic embedding.
- **Computer Vision (CV)**: Image generation and style transfer.
- **Deep Learning (DL)**: Neural network-based emotion and style analysis.
- **Neural Style Transfer (NST)**: Applying artistic styles to generated images.

## How It Works

1. **Input a story** – The system analyzes each sentence, extracts emotions, and selects an appropriate style.
2. **Generate images** – The pipeline produces a base image and applies a neural style transfer.
3. **View the final output** – Users can explore a visual representation of their story with analysis details in a Gradio UI.



## Usage

- First run the code as instructed in the notebook
- The system will analyze, generate, and stylize images based on the text you input in the UI.
- We can see the stylized image and original image and alongwith it some parameters on which model tries to decide the style.






