# [A Hybrid approach to longitudinal crack detection with computer vision, image processing, and machine learning]

## Overview
This project aims to detect longitudinal cracks during continuous casting by analyzing abnormal heat transfer between molten steel and the mold copper plate, caused by air gaps formed by these cracks. By integrating temporal and spatial crack data, we map this abnormal heat transfer onto a 2D plane. The approach employs image detection algorithms powered by the OpenCV library, complemented by machine learning models to minimize the false positive rate of the hybrid model. The primary goal is to reduce false positives while ensuring all crack instances are identified, offering a novel data-driven perspective for anomaly detection in the continuous casting process.

Link to this paper:https://link.springer.com/article/10.1007/s11663-025-03513-y

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)

## Installation
### Prerequisites
Before running the project, make sure you have the following dependencies installed:

- Python 3.6
- Required Python package including numpy, pandas, sklearn, matplotlib etc,.


## Usage
### Running the Model 
To run the model, follow these steps:

1. Download all files from this repository and ensure they are placed within the same root directory. Specifically, the folder "MeiSteel Longitudinal crack detection" must be located in the same root directory as the other code files.
2. Open a terminal, navigate to the directory containing the code, and execute the script by running:
python MeiSteel_Crack_recognition_by_combined_machine_learning_and_image_processing_method.py
3. Once the script runs, intermediate results—such as confusion matrices—will appear in pop-up windows. To disable these, comment out the cv.imshow() function calls in the code.
4. After execution, a folder named Meisteel_image_detection_result will be generated, containing the longitudinal crack detection results.

## Project Structure
The project is organized as follows:

│  main.py
│  MeiSteel_Crack_recognition_by_combined_machine_learning_and_image_processing_method.py
│  Visualization_of_detection_result.py
├─MeiSteel Longitudinal crack detection
│  ├─Crack

## Models
### Model Used

The project utilizes the following machine learning models:

Logistic regression (linear)

SVM (linearn and non-linear)

Random forest (ensemble model)

Adaboost (ensemble model)

These four machine learning models are employed to distinguish between true and false longitudinal cracks, addressing the high number of false positives that persist after applying a pure image detection algorithm to thermographic images of longitudinal cracks.

## Results
The following is a summary of the testing results obtained:

Accuracy: 0.94

Precision: 0.64

Recall: 1.00

F1 score: 0.78

TPR: 100.00%

FPR: 7.27%

This hybrid model, which integrates an image detection algorithm with SVM, delivers the best performance after evaluating its generalization performances against the three other models mentioned above.
