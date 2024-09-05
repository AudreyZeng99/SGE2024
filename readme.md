# Prototype-based Explanation for Embedding-based Link Prediction in Knowledge Graphs

before running the codes, here are the logic of the output:

run **"./src/trainer.py"** -> see output in ./exp/{data_name}/TranE/

run **"./src/tester.py"** -> see output in ./exp/{data_name}/test/score_{data_name}.json

run **"./src/explainer.py"** / "example_explanations.py" / "global_explanation.py" -> see output in ./exp/{data_name}/explanations/


Go to `./src/`, Run the code following the process:

### Example for the reimplement of TransE on WN18RR dataset.
#### Train TransE on WN18RR
```commandline
python trainer.py --data_name wn18rr
```
#### Get the original Test score of TransE on WN18RR
```commandline
python tester.py --data_name wn18rr
```

#### Get Single-Explanation for each candidate prediction
```commandline
python explainer.py --data_name wn18rr
python example_explanations.py --data_name wn18rr
```


#### Get Global-Explanation for TransE on WN18RR
```commandline
python global_explanation.py --data_name wn18rr --visible_range 3 # you can set how many explanation you want to see by setting '--visible_range'
```


