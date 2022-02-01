#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:39:35 2022

@author: alessandro
"""

import sys
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes#1
from skmultiflow.lazy import KNNClassifier#2
from skmultiflow.trees import HoeffdingTreeClassifier#3
from skmultiflow.meta import AdaptiveRandomForestClassifier#4
from skmultiflow.neural_networks import PerceptronMask#5
from skmultiflow.rules import VeryFastDecisionRulesClassifier#6
from skmultiflow.prototype import RobustSoftLearningVectorQuantization#7

dataStreamPath = '/Users/alessandro/Downloads/dati/'
filename='01Susy.csv'
stream = FileStream(dataStreamPath + filename)

# 2. Instantiate the HoeffdingTreeClassifier
knn = KNNClassifier()#2
nb = NaiveBayes()#1
rslvq = RobustSoftLearningVectorQuantization()#7
ht = HoeffdingTreeClassifier()#3
arf = AdaptiveRandomForestClassifier()#4
perceptron = PerceptronMask()#5
learner = VeryFastDecisionRulesClassifier()#6


samplexbatch = 55600
sys.stdout = open("output01.txt", "w")
while samplexbatch >=13900:
  
# 3. Setup the evaluator
  evaluator = EvaluatePrequential(batch_size=int(samplexbatch), pretrain_size=int(samplexbatch),
  metrics=['accuracy', 'kappa','precision','recall','f1', 'true_vs_predicted', 'running_time'],
  max_samples=1000800,
  output_file='/Users/alessandro/Tesi/results01.csv')
  evaluator.evaluate(stream=stream, model=[ht,knn,nb,rslvq,arf,perceptron,learner], model_names=['HT','KNN','NB','RSLVQ','ARF','PERCEPTRON','LEARNER'])
  samplexbatch = samplexbatch/2

sys.stdout.close()  