import json
import pandas
import pandas as pd   
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt 

path = '/storage-nlp/nlp/hoangnv74/legaleval/method2/results/2023-01-10_09_46_37_pubmed-20k_baseline/'

result = pd.read_json(path_or_buf=path+'0_0_results.jsonl', lines=True)

GEN_LABELS = ["PREAMBLE", "NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS", "PRE_RELIED",
              "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
plt.figure(figsize = (10,7))
confusion = result.iloc[-7]['dev_confusion'][2:]
confusion = [t[2:] for t in confusion]
ax = sn.heatmap(confusion, annot=True, xticklabels=GEN_LABELS, yticklabels=GEN_LABELS)

plt.show()
plt.savefig(f'{path}confusion_matrix.png')