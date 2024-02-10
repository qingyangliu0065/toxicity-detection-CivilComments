## project vision

Spurious correlation is an undesirable phenomenon prevalent in contemporary Natural Language Processing (NLP) tasks. Statistically, it is defined as any correlation between two variables that appears to be causal but is actually not. The existence of spurious correlation can be dangerous in commercial NLP models, as the model concludes false correlation to predict the seemingly right results on the training set, then triggers abnormal performance on the test sets by relying too much on this spurious relationship between specific attributes and the label.

In our work, we try to mitigate the effects of spurious correlation in toxic online comment detection. Given an input text of a comment x ∈ X, we want to classify whether the comment is toxic Y = {toxic; non-toxic}, which is spuriously correlated with the mentions of the demographic identities A = {identity, no identity}. Figure 1 shows sample sentences from the dataset illustrating this definition. Here demographic identity refers to a set of 8 sensitive identities including male, female, LGBTQ, Christian, Muslim, other religions, Black, and White.

We propose a revised two-stage training pipeline to im- improve the model’s capacity in learning the under-represented groups and improve the worst group performance, and we also implement a simple yet efficient new model structure by appending a biLSTM layer to a pre-trained BERT model. We arrive at two main conclusions:
1. By deploying a more advanced group-balancing technique, we achieve a noticeable increase in the worst group accuracy.
2. By appending a biLSTM layer to a pre-trained BERT model, we improve the worst group performance specifically on input texts with the issue of long-term dependencies.

## Instruction for reproducibility

To download the data, please follow the instructions according to https://github.com/anniesch/jtt. 

This project involves extensive coding, including the engineering in coding the basic framework,
which takes enormous amount of time. 

To run the experiment for AUX, please enter 

```
python run_metaScript.py --dataset jigsaw --jtt --stageOne_T 0 --up_weights 4 --stageOne_epochs 3 --stageTwo_epochs 4 --subsample_non_error 1.0 0.8 0.5 0.3 (--bertLstm)
```

To run the experiment for JTT, please enter

```
python run_metaScript.py --dataset jigsaw --jtt --stageOne_T 0 --up_weights 4 --stageOne_epochs 3 --stageTwo_epochs 4 --subsample_non_error 1.0 0.8 0.5 0.3 (--bertLstm)
```

To perform the length analysis, please enter 

```
python length_analysis.py --model_path xxx
```

where model_path is the address for the model.
