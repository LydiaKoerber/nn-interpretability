# nn-interpretability

## Usage

### DistilBERT Attributions
- fine-tune the model yourself: `models/model.py`
- download the fine-tuned model from [Google Drive](https://drive.google.com/file/d/196sLJymyrjd0io00H3G5P-d9e3PE-8hr/view?usp=sharing) and save it to the `models/` directory; you may need to adjust the file path in `captum-explain.py` to run the attribution computations
- or analyse the attributions in `outputs/distilbert/`