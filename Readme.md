# Detection of Emotions by Text Analysis Using Machine Learning

## Team Members
- **Vivekanand Reddy Malipatel** (A20524971)
- **Chethan Harinath** (A20526469)
- **Divyansh Soni** (A20517331)

## Project Overview
This project aims to enhance emotion detection in textual data through the use of pre-trained Large Language Models (LLMs) via transfer learning. Traditional sentiment analysis models often fall short when it comes to understanding contextual nuances and cultural differences in human emotions. By fine-tuning LLMs on multilingual datasets, we strive to improve the models' comprehension of context and cultural variations, thereby boosting emotion detection across varied datasets and languages. Our comprehensive methodology includes literature review, model selection, data collection, preprocessing, model training, fine-tuning, and evaluation. The results underscore the effectiveness of pre-trained LLMs in capturing emotional nuances and their versatility across different languages.

## Additional Resources
For more details, including in-depth findings and analyses, refer to the project report available in this repository.

## Model Repositories
- **Base LLM**: [HuggingFace Repository](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- **Fine-tuned LLM**: [HuggingFace Repository](https://huggingface.co/VivekMalipatel23/mDeBERTa-v3-base-text-emotion-classification)

## Setup and Installation
1. Create a new conda environment.
2. Install the required dependencies from `pip_requirements.txt`. Note: Install Pytorch based on your system's configuration.
3. Download and place the Base LLM and Fine-tuned LLM models in the project folder.
4. Extract the Naive Bayes Model from `Baseline_Models/Naive_Bayes/NaiveBayes_model_files/compressed_model_files.zip` for inference use.

## Repository Structure

### `Dataset` Directory
- `text.csv`: Raw English dataset sourced from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data).
- `pre_process_dataset.py`: Script to preprocess the raw dataset. Modify source and target filenames as necessary.
- `data_analysis.py`: Script for exploratory data analysis. Update the source filename accordingly.
- `split_dataset.py`: Script to divide the dataset into training and testing sets.
- `label_compressor.py`: Script to compress labels for new datasets to be tested on our models.
- `Training_dataset/pre_processed_text_train_partition.csv`: Training dataset in English.
- `Testing_dataset/pre_processed_text_test_partition.csv`: Testing dataset in English.
- `translate.py`: Script to translate the English test dataset into other languages.
- `Testing_dataset/pre_processed_text_test_partition_es_translated.csv`: Testing dataset in Spanish.

### `Baseline_Models` Directory

#### LSTM
- `LSTM.py`: Python script for training the LSTM model.
- `LSTM_DDP.py`: Script for training LSTM in a Distributed CUDA GPU setup.
- `LSTM_model_files`: Directory containing the trained LSTM model.

#### Naive Bayes
- `naive_bayes.py`: Python script to train the Naive Bayes model.
- `NaiveBayes_model_files`: Directory containing the trained Naive Bayes model.

### Scripts
- `finetune_llm.py`: LLM trainer script. Adjust source and target filenames as needed.
- `compare_models.py`: Main script to test all models with the test dataset after training is complete.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
