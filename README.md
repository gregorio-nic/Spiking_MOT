# Spiking Multi-Omics Transformer (MOT) Model for Pan-Cancer Classification

## Introduction

The advent of high-throughput techniques has generated vast and diverse omics datasets, including genomics, transcriptomics, proteomics, metabolomics, and lipidomics. These datasets provide new opportunities for personalized medicine, allowing for a deeper understanding of patients' conditions. Traditionally, research has focused on single-omics studies, but there is a growing trend towards multi-omics approaches. Integrating multiple omics types offers a more comprehensive view, particularly in the study of complex diseases such as cancer, central nervous system disorders, and cardiovascular diseases.

This project leverages the Multi-Omics Transformer (MOT) architecture to classify multi-omics data. In addition, we explore the transformation of the MOT architecture into a spiking neural network, which aims to mimic the way neurons process information in the brain, to evaluate its performance compared to the traditional model.

## Dataset Overview

The dataset used in this project is the TCGA pan-cancer dataset, which is publicly available on the UCSC Xena data portal. This dataset includes samples from 33 different tumor types and incorporates five distinct omics data types:

1. mRNA (RNA-Seq gene expression): Gene expression profiles with 20,532 gene identifiers, normalized through a log2 transformation.
2. DNA Methylation: Data derived from the Illumina Infinium Human Methylation BeadChip arrays, with 485,578 probes. (Not used in this project)
3. Copy Number Variations (CNVs): Profiles containing 24,776 identifiers representing various copy number alterations.
3. miRNA: A dataset consisting of 743 identifiers, also log2-transformed for normalization.
4. Protein Expression: Comprising 210 identifiers related to protein expression levels.

One of the primary challenges with this dataset, common to most omics data, is the imbalance in sample numbers across different tumor types. For instance, breast cancer is represented by over 1,200 samples, whereas cholangiocarcinoma has fewer than 50 samples.

## Code structure
Here we present the most important code files with descriptions:

- src/multiomic_modeling
    - data_hdf5: folder used to contain dataset hdf5 file, initially empty then filled from the zip file.
    - artifacts: folder used to save checkpoints during training.
    - models
        - models.py: file containing base models.
        - base.py: base trainer's configuration (Wandb).
        - encoder.py: encoder module.
        - decoder.py: decoder module.
        - snn_transformer.py: implementation of snn version of transformer.py by pytorch.
        - trainer.py: implementation of snn training of the model.
    - loss_and_metrics.py: file for metrics computation.
    - neurobenchOmics: plugin implementation of neurobench for omics.  

## How to use the code <a target="_blank" href="https://colab.research.google.com/github/MLinApp-polito/mla-prj-24-mla24-prj21-gu1/blob/main/MLProject.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
The first step is to fork the repository into your own project.


Wandb implementation:
This isn't mandatory, but it allows real-time logging with weights and biases framework for loss and accuracy. To enable Wandb go in src->models->base.py and proceed with the following instruction:
1. uncomment from lines 120-131.
2. uncomment line 188.
3. uncomment line 257.

Following the google colab file:
1. Environment Setup: cloning the repository and installing the requirements.
2. Prepare the dataset.
3. Wandb: if enabled, enter API_key.
4. Imports and model params: importing the required libraries and setting the model parameters.
5. Download checkpoints and training: the training is started, possibly from a checkpoint.
6. Score and Test with neurobench: this is done in order to evaluate the model's performances.
7. Plots: same as before, we plot the neurons' activity.

## Future Directions

The next steps involve optimizing the integration of different omics types and expanding the model’s capability to handle more complex and heterogeneous datasets. Further research will focus on enhancing the spiking neural network architecture to improve its real-time processing capabilities. Other future works are:
- Increment model configuration settings.
- Explore different hyperparameters sets.
- Use more datasets to enanche the model capabilities.
