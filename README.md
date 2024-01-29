# VTCC-NLP-at-SemEval-2023-Task-6
This repository contains the source code for the models used for VTCC-NLP team's submission for SemEval-2023 Task 6 “Rhetorical Roles Prediction”. The model is described in the paper "VTCC-NLP at SemEval-2023 Task 6:Long-Text Representation Based on Graph Neural Network for Rhetorical Roles Prediction".
## Install dependencies
```
  pip install -r requirements.txt
```
## How to run the code
For training, you can run as the follows:
```
  bash lsp.sh
  bash train.sh
```
For inference, run this command:
```
python infer_new_graph.py custom_processed_input.json output_json_path model_path
```
