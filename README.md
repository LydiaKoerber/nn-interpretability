# nn-interpretability

This repository contains the code to a university project on interpretability measures in text classification.
It investigates and compares interpretability measures for Support Vector Machine and transformer-based (DistilBERT) models for text classification on the [20-newsgroups](http://qwone.com/~jason/20Newsgroups/) corpus.

## Usage

### Dataset
- data set preprocessing (excluding headers, footers, quotes and truncating) is done in `dataset/dataset.py`


### SVM

### DistilBERT Attributions
- fine-tune the model yourself: `models/model.py`
- download the fine-tuned model from [Google Drive](https://drive.google.com/file/d/196sLJymyrjd0io00H3G5P-d9e3PE-8hr/view?usp=sharing) and save it to the `models/` directory; you may need to adjust the file path in `captum-explain.py` to run the attribution computations
- or analyse the attributions in `outputs/distilbert/`

### Analyses
Analyses are conducted in 3 different Notebooks:
- `analysis-distilbert.ipynb` contains the analyses of the DistilBERT attributions
- `analysis-predictions.ipynb` compares the scores for the test set instances (SVM coefficients in `outputs/coefs_test.csv` and DistilBERT attributions in `outputs/distilbert_attributions.csv`), creates some visualizations of specific instances in `outputs/viz`
- `analysis-vocabs.ipynb`

## Authors
Lydia Körber and Lisanne Rüh