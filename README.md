# Information Retrieval and Text Summarization Training Script

This script trains a model for information retrieval or text summarization via command line parameters. It utilizes the BERT and BART models from the transformers library for these tasks and processes datasets using the datasets library.

## Parameter Description

### --task
- **Type**: str
- **Options**: ['retrieval', 'summarization']
- **Description**: Select the type of task to be performed. 'retrieval' is used for information retrieval, and 'summarization' is used for text summarization.

### --model_name_or_path
- **Type**: str
- **Default**: None
- **Description**: Specify the name or path of the pre-trained model. If not provided, the default model will be used: `bert-base-uncased` for information retrieval, `facebook/bart-large` for text summarization.

### --output_dir
- **Type**: str
- **Default value**: `./results`
- **Description**: Output directory for training results and models.

### --num_train_epochs
- **Type**: int
- **Default**: 3
- **Description**: The number of training iterations.

### --batch_size
- **Type**: int
- **Default value**: 16
- **Description**: The size of the training batch for each device.

## Script Functions

### Parsing Command Line Arguments
Parses the command line arguments using the `argparse` library and stores them in the `args` object.

### Load Pre-trained Models and Tokenizers
Loads pre-trained models and tokenizers according to task type (`--task`) and model name (`--model_name_or_path`).

### Loading the Dataset
- For the information retrieval task, load the `ms_marco` dataset.
- For the text summarization task, load the `samsum` dataset.

### Preprocessing Data
Defines the `preprocess_function` to convert raw data into an input format acceptable to the model. It uses the `dataset.map` method to apply the preprocessing function.

### Training the Model
Configures `TrainingArguments` to set training parameters such as learning rate, batch size, number of training epochs, etc. Trains the model using the `Trainer` class.

### Evaluate the Model
For the text summarization task, load the ROUGE metric and evaluate the model's performance on the validation set.

## Example Commands

```bash
python3 train.py --task retrieval --model_name_or_path bert-base-uncased --output_dir ./results --num_train_epochs 3 --batch_size 16

python3 train.py --task summarization --model_name_or_path facebook/bart-large --output_dir ./results --num_train_epochs 3 --batch_size 16
```

## Model and Parameter Selection
The two chosen models can serve as benchmark models in research and production.

### BERT (Bidirectional Encoder Representations from Transformers)

#### Reason for Choosing:
BERT is an encoder-only model that excels in a variety of natural language processing tasks, such as text classification. Its bi-directional encoder allows it to better understand context, making it particularly effective for information retrieval, query processing, and document matching.

#### Default Model:
- `bert-base-uncased` is a commonly used version for most English text processing tasks.

### BART (Bidirectional and Auto-Regressive Transformers)

#### Reason for Choosing:
BART is a sequence-to-sequence model that performs exceptionally well on various natural language processing tasks, particularly text generation, where it surpasses the BERT model.

### Default Model:
- `facebook/bart-large` is a commonly used version for most English text summarization tasks.

### Training Parameters
The training parameters are chosen based on GPU resources and the recommended settings from the model documentation. Advanced parameter adjustments can be made as needed to optimize performance. With these choices of parameters and models, the training process can be configured flexibly for specific tasks, resulting in an efficient and effective natural language processing model.



