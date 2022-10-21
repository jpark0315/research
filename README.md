# research

Personal Resarch Stuff For Curiosity's Sake

Used BNNs for model based policy optimization with SAC algorithm https://arxiv.org/abs/1801.01290 as a base learner. 

Found that BNNs does indeed handles aleatoric uncertainty better, but often converges poorly on tasks with large state spaces. Probably because 
the probability inference computations does not scale well, 
