# -*- coding: utf-8 -*-
#%% Import statements
from __future__ import division
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Main code is at the end of the file as this was written in spyder and functions
# need to be defined before they are called

#%% def create_sign
def create_sign(label):
    
    # add more or statements for different labels. 
    # For example, nltk's movie reviews use "pos" and "neg"
    # instead of 1 and 0
    if label == "pos" or label == str(1): 
        return 1
    else:
        return -1
        
    
#%% def initialise_weights
def initialise_weights(weights, reviews):
    
    # make list of all reviews
    review_list = [review[0] for review in reviews]

    # extract all unigrams from reviews
    unigrams = list(set([unigram for review in review_list for unigram in review]))
        
    for unigram in unigrams:
        weights['UNI: ' + str(unigram)] = 0
            
    return weights
    
    
#%% def create_features
def create_features(words):
    
    # create map phi
    phi = {}
    
    # count unigram occurrences
    for word in words:
        unigram = 'UNI: ' + str(word)
        
        if unigram in phi:
            phi[unigram] += 1
        else:
            phi[unigram] = 1
    
    return phi


#%% def predict_one
def predict_one(weights, phi):
    
    score = 0
    
    # iterate through phi
    for feature, value in phi.iteritems():
        
        if feature in weights:
            
            score += value * weights[feature]
            
    if score >= 0:    
        return 1
    else:
        return -1
    
    
#%% def update_weights
def update_weights(weights, phi, sign):    
    
    for feature, value in phi.iteritems():
        
        weights[feature] += value * sign
        
    return weights        


#%% def learn_weights
def learn_weights(data, iterations):

    # create map weights
    weights = {}
    
    # initialise weights with new features
    weights = initialise_weights(weights, data)        
            
    # loop a number of times defined in iterations
    for i in range(0, (iterations+1)):
    
        # for each labeled pair x, y (review, label) in the data
        for review in data:
            
            words = review[0]
            label = review[1]
            sign = create_sign(label) # convert dataset labels to +1/-1
    
            # create features from review line
            phi = create_features(words)
            
            # predict one
            sign_prime = predict_one(weights, phi)
            
            # update weights
            if sign_prime != sign:
                
                weights = update_weights(weights, phi, sign)
            
    return weights


#%% def predict_all
def predict_all(test_data, weights):
    
    predictions = []
    
    for review in test_data:
        
        words = review[0]
        label = review[1]
        sign = create_sign(label)        
        
        # create features
        phi = create_features(words)
        
        # predict one
        sign_prime = predict_one(weights, phi)   
        
        # add predictions to a list of text, given label and predicted label
        predictions.append((words, sign, sign_prime))
        
    return predictions

#%% def predict_all_unmarked
def predict_all_unmarked(test_data, weights):
    
    predictions = []
    
    for review in test_data:
        
        words = review      
        
        # create features
        phi = create_features(words)
        
        # predict one
        sign_prime = predict_one(weights, phi)   

        predictions.append((words, sign_prime))
        
    return predictions
    

#%% summarise_predictions
def summarise_predictions(predictions):
    
    right = 0
    wrong = 0
    total = 0
    
    for prediction in predictions:
        
        total += 1
        sign = prediction[1]
        sign_prime = prediction[2]
        
        if sign == sign_prime:
            right += 1
        else:
            wrong += 1
            
    # the percentage of the correctly predicted instances to the total, i.e. #correctly_predicted / #total
    accuracy = 100*(right / total)
    
    print "Right: " + str(right)
    print "Wrong: " + str(wrong)
    print "Accuracy: {0:.2f}%".format(accuracy) # display up to two decimals
    

#%% split_data
def split_data(data):
    
    count = 0
    
    new_total_positives = 3224
    new_total_negatives = 2444
    
    training_data = []
    test_data = []
    
    # add 3224 positives to the new training data set
    for line in data:
        while count < new_total_positives:
            count += 1
            popped_element = data.pop(0) # pop first        
            training_data.append(popped_element)
        # at this point, I have 7086 - 3224  = 3862 lines remaining, of which
        # 3862 - 3091 = 771 are remaining positives
        # confirm numbers by running count_data
    
    # knowing this I move on to grave the negatives
    count = 0    
    
    for line in data:
        while count < new_total_negatives:            
            count += 1
            popped_element = data.pop() # pop last
            training_data.append(popped_element)
        # I should now have 3091 - 2444 = 647 remaining negatives
        
    test_data = data
        
    '''
    print "\nfinal count = " + str(count)
    print "\nold data"
    count_data(data)
    print "\nnew data"
    count_data(training_data)    
        
    count_data(test_data)
    '''
    return training_data, test_data

#%% def count_data, use this to confirm numbers when splitting the dataset
def count_data(data):
    count1 = 0
    count0 = 0
    print "dataset size = " + str(len(data))
    for line in data:
        if line[1] == str(1):
            count1 += 1
        else:
            count0 += 1
    print "number of positive reviews = " + str(count1)
    print "number of negative reviews = " + str(count0)
'''
dataset size = 7086
number of positive reviews = 3995 (56.37%)
number of negative reviews = 3091 (42.62%)
80% of  total data  = 0.80 * 7086 = 5668.8 -> this is the size of the new training data, rounded down to 5668
Of the new training data I want 
56.37% positives: 5668 * 0.5637 = 3195 + 29 = 3224 (56.38%)
42.62% negatives: 5668 * 0.4262 = 2415 + 29 = 2444 (43.11%)
'''

#%% def export_results
def export_results(results, unmarked_test_data):
    
    results_file = open('results.txt', 'w')
    
    for predicted_line, unmarked_line in zip(results, unmarked_test_data):
        
        results_file.write(str(predicted_line[1]) + "\t" + str(unmarked_line))

#%% main

# set up stopwords list
stop_words = set(stopwords.words('english'))

# set up empty data lists
original_training_data = []
unmarked_test_data = []
pure_unmarked_test_data = []
# chose number of weights training iterations
ITERATIONS = 26 # 16 was found to return the best accuracy

# parse data
with open('training.txt') as f:
    for line in f:
        training_data_line = line.split("\t") # list of [score, text]
        line_text = training_data_line[1]
                
        # tokenize the text
        tokenized_sentences = word_tokenize(line_text)
        # convert each token to lower case and strip it of punctuation. Remove english stop words 
        # using the nltk stop words list. Also remove punctuation using Python's string.punctuation.
        tokenized_sentences = [
                w.lower().strip(string.punctuation) 
                for w in tokenized_sentences 
                if w.lower() not in stop_words and w.lower() not in string.punctuation
                ]
        # add tokenized sentence and score values into data list
        original_training_data.append((tokenized_sentences, training_data_line[0]))

# split data into training and test data (80:20 split) to be able to do holdout cross-validation 
training_data, test_data = split_data(original_training_data)                

weights = learn_weights(training_data, ITERATIONS)

'''
# Uncomment to test accuracy of model on marked data
iterations loop to create accuracy chart. See report
for i in range(2, 40, 2):
    print "\niterations: " + str(i)      
    # learn weights
    weights = learn_weights(training_data, i)
    
    # predict!

    predictions = predict_all(test_data, weights)
    summarise_predictions(predictions)
'''

'''
# Uncomment to run sentiment analysis in test file and return results file
# open and parse test data
with open('testdata.txt') as f:
    for line in f:
        test_data_line = line.split("\t") # list of [score, text]
        line_text = str(training_data_line)
        # tokenize the text
        tokenized_sentences = word_tokenize(line_text)
        # convert each token to lower case and strip it of punctuation. Remove english stop words 
        # using the nltk stop words list. Also remove punctuation using Python's string.punctuation.
        tokenized_sentences = [
                w.lower().strip(string.punctuation) 
                for w in tokenized_sentences 
                if w.lower() not in stop_words and w.lower() not in string.punctuation
                ]
        # add tokenized sentence and score values into data list
        unmarked_test_data.append(tokenized_sentences)

# open again but keep original formatting to create a more readable results file     
with open('testdata.txt') as f:
    for line in f:
        pure_unmarked_test_data.append(line)

results = predict_all_unmarked(unmarked_test_data, weights)
export_results(results, pure_unmarked_test_data)
'''