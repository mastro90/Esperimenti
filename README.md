# Robustness Analysis of Data Stream Algorithms

## Project Overview
This project, developed within the framework of **Intelligent Systems for Digital Communication**, focuses on the **robustness analysis of various data stream algorithms**.  
The main objective is to evaluate whether the **batch size** used during processing positively or negatively affects the classification performance.  
A specific **case study on patients affected by bipolar disorder** was used as a primary dataset.

## Context
**Bipolar disorder** is a chronic mental illness characterized by alternating depressive and manic phases.  
The data related to bipolar patients are highly dynamic and present sudden variations, making their automatic analysis particularly challenging for machine learning algorithms.

## Data Stream Analysis
**Data stream analysis** aims to extract knowledge from large, continuous, and potentially infinite data flows.  
Key characteristics include:
- **Continuous flow of data**  
- **Concept drifting** – the data distribution changes over time  
- **Data volatility** – incoming data are not permanently stored  

To handle these challenges, **incremental and online learning algorithms** were applied.

## Tools and Framework
The implementation was carried out using **Scikit-Multiflow**, an open-source machine learning framework for data streams, inspired by:
- **MOA (Massive Online Analysis)**  
- **Scikit-Learn**

### Evaluation Methods
Two main evaluation strategies were considered:
- **Hold-out evaluation**: testing with predefined test sets  
- **Prequential evaluation**: each new data instance is first used for testing, then for training

## Experimental Setup

### Algorithms
The following data stream classification algorithms were analyzed:
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Hoeffding Tree  
- Adaptive Random Forest  
- Robust Soft Learning Vector Quantization  
- Very Fast Decision Rules  
- Perceptron  

### Evaluation Types
- **Prequential Train-Test Evaluation**

### Performance Metrics
- Precision  
- Recall  
- Accuracy  
- F1-score  
- Kappa score  

### Datasets
- **Bipolar** (main case study)  
- **KDDCUP**, **RLCPS**, **SUSY**, **HEPMASS**, **HIGGS** (benchmark datasets)

## Future Work
- Development of **new models** providing a trade-off between accuracy and execution time  
- **Testing on more powerful hardware** to process larger data volumes efficiently  
