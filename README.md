Brief
Trains a model for information retrieval or text summarization via command line parameters. Here I use the BERT and BART models from the transformers library for these two tasks respectively, and load and process the dataset via the datasets library.

Parameter description
--task
Type: str
Options: [‘retrieval’, ‘summarization’]
Description: Select the type of task to be performed. retrieval is used for information retrieval, summarisation is used for text summarization.

---model_name_or_path
Type: str
Default: None
Description: Specify the name or path of the pre-trained model. If not provided, the default model will be used: bert-base-uncased for information retrieval, facebook/bart-large for text summarisation.

--output_dir
Type: str
Default value: . /results
Description: Output directory for training results and models.

--num_train_epochs
Type: int
Default: 3
Description: The number of training iterations.

---batch_size
Type: int
Default value: 16
Description: The size of the training batch for each device.

Script Functions

Parsing command line arguments
Parses the above command line arguments using the argparse library and stores them in the args object.

Load pre-trained models and splitters
Loads pre-trained models and splitters according to task type (--task) and model name (--model_name_or_path).

Loading the dataset
For the information retrieval task, load the ms_marco dataset.
For the text summarisation task, load the samsum dataset.

Preprocessing data
Define the preprocess_function function that converts raw data into an input format acceptable to the model.
Use the dataset.map method to apply the preprocess function.

Training the model
Configure TrainingArguments to set training parameters such as learning rate, batch size, number of training rounds, etc.
Train the model using the Trainer class.

Evaluate the model
For the text summarisation task, load the ROUGE metric and evaluate the model's performance on the validation set.

Command
python3 train.py --task retrieval --model_name_or_path bert-base-uncased --output_dir . /results --num_train_epochs 3 --batch_size 16

python3 train.py --task summarisation --model_name_or_path facebook/bart-large --output_dir . /results --num_train_epochs 3 --batch_size 16


Reasons for model and parameter selection
BERT (Bidirectional Encoder Representations from Transformers)
Reason for choosing: The BERT model is a encoder-only model and performs well on a variety of natural language processing tasks like text classification. For information retrieval, BERT's bi-directional encoder allows it to better understand context and thus excel at processing queries and document matching.
Default model: bert-base-uncased is a commonly used version for most English text processing tasks.
BART（Bidirectional and Auto-Regressive Transformers）
Reason for choosing: The BART model is a sequence2sequence model and performs well on a variety of natural language processing tasks like text generartion than the BERT model. 
Default model: facebook/bart-large is a commonly used version for most English text processing tasks.
Both chosen models can be used as benchmark models in research and production.

The training parameters are chosen based on GPU resource and reccomanded parameters from their documentations. Further advanced adjustments of parameters could be added if needed. With these choices of parameters and models, the training process can be flexibly configured to the specific task, resulting in an efficient and effective natural language processing model.
