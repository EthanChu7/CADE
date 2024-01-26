import deeprobust.image.netmodels.train_model as trainmodel

trainmodel.train('vgg16', 'pendulum', 'cuda', 200, lr=1e-4)