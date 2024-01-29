# VTCC-NLP-at-SemEval-2023-Task-6
This repository contains the source code for VTCC-NLP team's method for SemEval-2023 Task 6 “Rhetorical Roles Prediction”. Our method achieves the top 4 in the public leaderboard of the sub-task B. 
The model is described in the paper "VTCC-NLP at SemEval-2023 Task 6:Long-Text Representation Based on Graph Neural Network for Rhetorical Roles Prediction".
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
