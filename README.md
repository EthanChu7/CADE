# Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples

This repository is the official implementation of our paper [Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples](https://arxiv.org/abs/2312.13628).
In this work we propose CADE, a framework that can generate Counterfactual ADversarial Examples.

![framework of CADE.](framework.png)

## Example for latent
```
attacker = CADELatent(generative_model=generative_model, 
                      attacking_nodes=attacking_nodes, 
                      substitute=substitute, 
                      device=device).to(device)
                      
x_cade = attacker.attack_whitebox(x=x, 
                                  label=label, 
                                  lr=step_size, 
                                  epochs=num_steps, 
                                  epsilon=epsilon,
                                  causal_layer=causal_layer
                                  )                  
```


## Example for observable
```
attacker = CADEObservable(causal_dag, 
                          attacking_nodes=np.array(l_attacking_nodes[mode]), 
                          y_index=y_index, 
                          substitute=model_base)
                          
x_cade = attacker.attack(endogenous=endogenous, 
                         epsilon=epsilon, 
                         causal_layer=causal_layer,
                         num_steps=num_steps,
                         step_size=step_size
                         )                                                   
```

## Experiment on Pendulum
### To reproduce the result on Pendulum, run 
```
python -u Experiment_CADE_Pendulum.py --substitute resnet50
```

```
python -u Experiment_CADE_Pendulum.py --substitute vgg16
```


## Experiment on CelebA
### To reproduce the result on CelebA, run 
```
python -u Experiment_CADE_CelebA.py --substitute resnet50
```

```
python -u Experiment_CADE_CelebA.py --substitute vgg16
```

## Experiment on SynMeasurement
### To reproduce the result on SynMeasurement, run 
```
python -u Experiment_CADE_SynMeasurement.py --substitute lin
```

```
python -u Experiment_CADE_SynMeasurement.py --substitute mlp
```

## Download the checkpoints for the experiment, please visit:

## Feel free to train the models from scratch, run

model_training/Pendulum/model_train_standard.py

model_training/Pendulum/model_train_adversarial.py
