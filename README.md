# Shakespeare Fine-Tuning Repository

Welcome to the Shakespeare Fine-Tuning Repository! This repository contains code and resources for fine-tuning a language model on Shakespearean texts.

## Overview

This project aims to fine-tune a pre-trained language model using Shakespearean texts to generate text in the style of William Shakespeare. Fine-tuning allows the model to adapt its parameters to a specific domain, in this case, the language and style of Shakespeare's works.

## Model
We use the Hugging Face Transformers library for fine-tuning a pre-trained gpt-2 language model. 

## Requirements

To run the code in this repository, you'll need to install the required dependencies using:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:  In *lm_fine_tuning.py* were used tools from HuggingFace library, to train the model, which requires .txt data. 
To convert our data from .csv to .txt, run **csv_to_text.py**:
    ```
    python csv_to_text.py
    ```

2. **Fine-Tuning**: Use the provided scripts to fine-tune a GPT-2 pre-trained language model on your Shakespearean data.

    ```bash
    python lm_fine_tuning.py
    ```

3. **Inference**: After fine-tuning, you can generate Shakespearean-style text using the following command:

    ```bash
    python example.py
    ```
    This will save the generated text to the path `data/generated`.

## Additional

In ```Shakespeare.ipynb``` you can find fine-tuning process of model using torch style

## Conclusion
Due to resource limitation, model requires further training, adjusting it's parameters, that is learning rate, batch size and epochs
No metric was used in codes. Although, I tried to add Perplexity metric, but unfortunately it will take additional time.