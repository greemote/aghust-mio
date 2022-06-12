AGH UST WFiIS  
metody inteligencji obliczeniowej  
topic: 8 - Predykcja zainteresowania postami w social media z użyciem metod NLP  
project group: Arkadiusz Trojanowski, Łukasz Kisielewski, Wiktor Gaworek  
language: MATLAB  
date: 15.06.2022  

prepareData.m should be used to generate the proper matrix from the .csv file for the net, though it takes a lot of time - better use already created data stored in preppedData.mat  
learn.m teaches the network and plots the results  
use analyzeResults.m to calculate the accuracy etc., open mostPopularWords.mat to load vocabulary words from 5 most popular tweets, open explainer.mat so you don't have to compute it again (cause it takes a while)
