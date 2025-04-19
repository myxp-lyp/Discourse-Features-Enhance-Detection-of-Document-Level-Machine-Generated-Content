# Discourse-Features-Enhance-Detection-of-Document-Level-Machine-Generated-Content
IJCNN official repo


This is readme of the above paper.

1. Refer to https://github.com/najoungkim/pdtb3/tree/master to train a PDTB pretrained model
2. Use PDTB_predict.py to store the data encoding (due to the memory constraints as its quite big when directly making DTransformer prediction)
3. Use DTransformer.py to predict the results
NOTE that transformer dependencies has to replace the original one as we have hierarchy level transformer

The writingprompts dataset are available in https://drive.google.com/file/d/1iQTjF2e2GcfBn6y6eNkSiD62Ce-p8xgN/view?usp=sharing as it is too large.

We also provided revised_text.txt, which contains 3 documents and 10 paragraphs specified in the paper. These were randomly selected from writingpromts test set. 
