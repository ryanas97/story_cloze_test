import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
from scipy.special import softmax


def scores_lists_multi(context_embs, ending_0_embs, ending_1_embs, model):
    """Returns a list of predictions based on model."""
    predicted_embs = model(context_embs) # return array [total_dataset, 768]
    score_list = []
    label = []

    for idx in range(predicted_embs.shape[0]):
        pred_emb = predicted_embs[idx, :]
        score_0 = np.dot(pred_emb, ending_0_embs[idx, :])
        score_1 = np.dot(pred_emb, ending_1_embs[idx, :])
        score_list.append(softmax([score_0, score_1]))
        label.append(int(score_0 < score_1))

    return score_list, label

from data import Data

data = { 'train' : Data('data/train2017.csv').get_train_data(),
            'valid_2016' : Data('data/valid2016.csv').get_validtest_data(),
            'valid_2018' : Data('data/valid2018.csv').get_validtest_data(),
            'test' : Data('data/test2016.csv').get_validtest_data() }

embedded_data = dict()

with open('dummy/train_sentencebert.pkl', 'rb') as f:
    load_data = pickle.load(f)
    embedded_data['train'] = {'context' : load_data['contexts'], 
                                'ending' : load_data['endings'] }

with open('dummy/valid_2016_sentencebert.pkl', 'rb') as f:
    load_data = pickle.load(f)
    embedded_data['valid_2016'] = { 'context' : load_data['contexts'],
                                    'ending0' : load_data['endings_0'], 
                                    'ending1' : load_data['endings_1'] }

with open('dummy/valid_2018_sentencebert.pkl', 'rb') as f:
    load_data = pickle.load(f)
    embedded_data['valid_2018'] = { 'context' : load_data['contexts'],
                                    'ending0' : load_data['endings_0'], 
                                    'ending1' : load_data['endings_1'] }

with open('dummy/test_2016_sentencebert.pkl', 'rb') as f:
    load_data = pickle.load(f)
    embedded_data['test'] = { 'context' : load_data['contexts'],
                                'ending0' : load_data['endings_0'], 
                                'ending1' : load_data['endings_1'] }

model = tf.keras.models.load_model('dummy/best_model_sbert.h5') 

scores, chosen_label = scores_lists_multi(embedded_data['test']['context'], embedded_data['test']['ending0'], embedded_data['test']['ending1'], model)
# print(scores, chosen_label)

ground_truth = [ex['label'] for ex in data['test']]
# print('ground_truth : ', ground_truth)

verdict = [x == y for x,y in zip(chosen_label, ground_truth)]
# print('verdict: ', verdict)
# print('result: ', sum(verdict)/len(verdict))

table = {'No': range(1, len(ground_truth)+1),
        'Probability Ending 1': [x[0] for x in scores],
        'Probability Ending 2': [x[1] for x in scores],
        'Ground Truth': [el+1 for el in ground_truth],
        'Verdict':verdict
        }

df = pd.DataFrame(table, columns = ['No', 'Probability Ending 1', 'Probability Ending 2', 'Ground Truth', 'Verdict'])
df = df.sort_values(by=['Verdict', 'Probability Ending 1'])
df.to_csv('tables/table_multi_bert.csv', sep='\t')
print(df)

print('----------------------------------------------------------------------------------------------------')

def scores_list_binary(context_embs, ending_0_embs, ending_1_embs, model):
    scores_ending_0 = model(tf.concat([context_embs, ending_0_embs], -1))
    # print("scores_ending_0 : ", np.array(scores_ending_0))
    scores_ending_1 = model(tf.concat([context_embs, ending_1_embs], -1))
    # print("scores_ending_1 : ", np.array(scores_ending_1))
    score_list = []
    for score0, score1 in zip(scores_ending_0[:, 1], scores_ending_1[:, 1]):
        score_list.append(softmax([score0, score1]))
    # print("scores : ", scores)
    label = [int(x < y) for x,y in zip(scores_ending_0[:, 1], scores_ending_1[:, 1])]
    # print('chosen_label : ', chosen_label)
    return score_list, label

model = tf.keras.models.load_model('dummy/best_model_sbert_2.h5') 

scores, chosen_label = scores_list_binary(embedded_data['test']['context'], embedded_data['test']['ending0'], embedded_data['test']['ending1'], model)
# print(scores, chosen_label)

ground_truth = [ex['label'] for ex in data['test']]
# print('ground_truth : ', ground_truth)

verdict = [x == y for x,y in zip(chosen_label, ground_truth)]
# print('verdict: ', verdict)
# print('result: ', sum(verdict)/len(verdict))

table = {'No': range(1, len(ground_truth)+1),
        'Probability Ending 1': [x[0] for x in scores],
        'Probability Ending 2': [x[1] for x in scores],
        'Ground Truth': [el+1 for el in ground_truth],
        'Verdict':verdict
        }

df = pd.DataFrame(table, columns = ['No', 'Probability Ending 1', 'Probability Ending 2', 'Ground Truth', 'Verdict'])
df = df.sort_values(by=['Verdict', 'Probability Ending 1'])
df.to_csv('tables/table_binary_bert.csv', sep='\t')
print(df)

df_90 = df.loc[(df['Probability Ending 1'] > 0.9) | (df['Probability Ending 2'] > 0.9)] 
df_90_true = df_90.loc[(df['Verdict'] == True)]
print(df_90_true.shape)
df_80 = df.loc[(df['Probability Ending 1'] > 0.8) | (df['Probability Ending 2'] > 0.8)] 
df_80_true = df_80.loc[(df['Verdict'] == True)]
print(df_80_true.shape)
df_70 = df.loc[(df['Probability Ending 1'] > 0.7) | (df['Probability Ending 2'] > 0.7)] 
df_70_true = df_70.loc[(df['Verdict'] == True)]
print(df_70_true.shape)
df_60 = df.loc[(df['Probability Ending 1'] > 0.6) | (df['Probability Ending 2'] > 0.6)] 
df_60_true = df_60.loc[(df['Verdict'] == True)]
print(df_60_true.shape)
df_50 = df.loc[(df['Probability Ending 1'] > 0.5) | (df['Probability Ending 2'] > 0.5)] 
df_50_true = df_50.loc[(df['Verdict'] == True)]
print(df_50_true.shape)

import matplotlib.pyplot as plt
 
y_bert = [ df_50_true.shape[0]-df_60_true.shape[0],
            df_60_true.shape[0]-df_70_true.shape[0],
            df_70_true.shape[0]-df_80_true.shape[0],
            df_80_true.shape[0]-df_90_true.shape[0],
            df_90_true.shape[0]
            ]

y_sbert = [ df_50_true.shape[0]-df_60_true.shape[0],
            df_60_true.shape[0]-df_70_true.shape[0],
            df_70_true.shape[0]-df_80_true.shape[0],
            df_80_true.shape[0]-df_90_true.shape[0],
            df_90_true.shape[0]
            ]

x = ['50-60', '60-70', '70-80', '80-90', ' 90-100']

fig, ax = plt.subplots(1, 1)
ax.grid()
ax.set_axisbelow(True)
bar_bert = ax.bar([el - 0.15 for el in range(len(y_bert))], y_bert, width=0.3, align='center', color='blue')
bar_sbert = ax.bar([el + 0.15 for el in range(len(y_sbert))], y_sbert, width=0.3, align='center', color='orange')
plt.title('Distribution of Correct Ending Score (BCM)')
plt.xlabel('Correct Ending Score (Out of 100)')
plt.ylabel('No. of correct answers')
plt.xticks(range(len(y_bert)), x, size='small')
plt.legend((bar_bert[0], bar_sbert[0]), ('BERT', 'S-BERT'))
plt.show()
plt.savefig('graphs/BCM.png')

print('------------------------------------------------------------')
