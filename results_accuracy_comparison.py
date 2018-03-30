#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:30:59 2018

@author: celebren
"""

results = []
checked_results = []
combo = []

with open('results.txt') as f:
    for line in f:
        data_line = line.split("\t") # list of [score, text]
        score = data_line[0]
        results.append(score)
        
with open('checked_results.txt') as f:
    for line in f:
        data_line = line.split("\t") # list of [score, text]
        score = data_line[0]
        checked_results.append(score)
        
print "new results: " + str(results[:10])
print "old results: " + str(checked_results[:10])

count = 0
mistakes = 0
for i in results:
    if i != checked_results[count]:
        mistakes += 1
    count += 1

print "mistakes: " + str(mistakes)