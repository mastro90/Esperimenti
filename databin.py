#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:05:35 2022

@author: alessandro
"""

import sys
import pandas as pd
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
filename='databin.csv'


dataSet = pd.read_csv(dataStreamPath + filename)
dataSet = dataSet.astype('int')# Cast 


dataSet.to_csv('/Users/alessandro/Tesi/file1.csv' ,index=False)
stream = FileStream('/Users/alessandro/Tesi/file1.csv')


knn = KNNClassifier()#2

nb = NaiveBayes()#1
rslvq = RobustSoftLearningVectorQuantization()#7
ht = HoeffdingTreeClassifier()#3
arf = AdaptiveRandomForestClassifier()#4
perceptron = PerceptronMask()#5
learner = VeryFastDecisionRulesClassifier()#6


samplexbatch = 36840
sys.stdout = open("databin.txt", "w")
while samplexbatch >=9210:
  
# # 3. Setup the evaluator
   evaluator = EvaluatePrequential(batch_size=int(samplexbatch), pretrain_size=int(samplexbatch),
   metrics=['accuracy', 'kappa','precision','recall','f1', 'true_vs_predicted', 'running_time'],
   max_samples=921000,
   output_file='/Users/alessandro/Tesi/resultsDataBin.csv')
   evaluator.evaluate(stream=stream, model=[knn,ht,nb,rslvq,arf,perceptron,learner], model_names=['KNN','HT','NB','RSLVQ','ARF','PERCEPTRON','LEARNER'])
   samplexbatch = samplexbatch/2
 
sys.stdout.close()  