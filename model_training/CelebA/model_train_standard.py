import deeprobust.image.netmodels.train_model as trainmodel

trainmodel.train('ResNet50', 'CelebA', 'cuda', 100, lr=0.01)