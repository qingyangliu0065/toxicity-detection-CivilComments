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
