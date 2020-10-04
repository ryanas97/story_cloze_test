import seaborn as sns
import matplotlib.pyplot as plt
from models import NNMultiClassModel, NNBinaryClassModel, compute_accuracy
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from scipy.special import softmax

class Result():

    def __init__(self, trainedModel, embeddedData, dataset, trainingData, typeNN, typeModel):
        self.data_graph = trainingData
        self.data_embd = embeddedData
        self.dataset = dataset
        self.type = typeNN
        self.model = trainedModel
        self.model_type = typeModel

    def create_graph_multi(self):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        x_train_step = self.data_graph['X_train_step']
        y_train_loss = self.data_graph['Y_VALUE']['train_loss']
        y_valid_2016_loss = self.data_graph['Y_VALUE']['valid_2016_loss']
        y_valid_2018_loss = self.data_graph['Y_VALUE']['valid_2018_loss']
        y_train_accuracy = self.data_graph['Y_VALUE']['train_accuracy']
        y_valid_2016_accuracy = self.data_graph['Y_VALUE']['valid_2016_accuracy']
        y_valid_2018_accuracy = self.data_graph['Y_VALUE']['valid_2018_accuracy']

        plt.figure(figsize=(10,10))
        plt.title("Loss Graph ("+ self.model_type.upper() +" + MCM)", fontsize=20)
        plt.xlabel("Training Step", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.plot(x_train_step, y_train_loss, label = 'train loss')
        plt.plot(x_train_step, [sum(x)/2 for x in zip(y_valid_2016_loss, y_valid_2018_loss)], label = 'validation loss')
        plt.legend(prop={'size': 15})
        # print('Average loss train : ', sum(y_train_loss) / len(y_train_loss))
        # print('Average loss val : ', sum([sum(x)/2 for x in zip(y_valid_2016_loss, y_valid_2018_loss)]) / len([sum(x)/2 for x in zip(y_valid_2016_loss, y_valid_2018_loss)]))
        plt.savefig('graphs/loss_'+ self.model_type +'_mcm.png')

        plt.figure(figsize=(10,10))
        plt.title("Accuracy Graph ("+ self.model_type.upper() +" + MCM)", fontsize=20)
        plt.xlabel("Training Step", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.plot(x_train_step, y_train_accuracy, label = 'train acc')
        plt.plot(x_train_step, [sum(x)/2 for x in zip(y_valid_2016_accuracy, y_valid_2018_accuracy)], label = 'validation acc')
        plt.legend(prop={'size': 15})
        # print('Average acc train : ', sum(y_train_accuracy) / len(y_train_accuracy))
        # print('Average acc val : ', sum([sum(x)/2 for x in zip(y_valid_2016_accuracy, y_valid_2018_accuracy)]) / len([sum(x)/2 for x in zip(y_valid_2016_accuracy, y_valid_2018_accuracy)]))
        plt.savefig('graphs/acc_'+ self.model_type +'_mcm.png')

    def create_graph_binary(self):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        x_train_step = self.data_graph['X_train_step']
        y_train_loss = self.data_graph['Y_VALUE']['train_loss']
        y_valid_loss = self.data_graph['Y_VALUE']['valid_loss']
        y_train_accuracy = self.data_graph['Y_VALUE']['train_accuracy']
        y_valid_accuracy = self.data_graph['Y_VALUE']['valid_accuracy']

        plt.figure(figsize=(10,10))
        plt.title("Loss Graph ("+ self.model_type.upper() +" + BCM)", fontsize=20)
        plt.xlabel("Training Step", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.plot(x_train_step ,y_train_loss, label = 'train loss')
        plt.plot(x_train_step, y_valid_loss, label = 'validation loss')
        plt.legend(prop={'size': 15})
        # print('Average loss train : ', sum(y_train_loss) / len(y_train_loss))
        # print('Average loss val : ', sum(y_valid_loss) / len(y_valid_loss))
        plt.savefig('graphs/loss_'+ self.model_type +'_bcm.png')

        plt.figure(figsize=(10,10))
        plt.title("Accuracy Graph ("+ self.model_type.upper() +" + BCM)", fontsize=20)
        plt.xlabel("Training Step", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.plot(x_train_step, y_train_accuracy, label = 'train acc')
        plt.plot(x_train_step, y_valid_accuracy, label = 'valid acc')
        plt.legend(prop={'size': 15})
        # print('Average acc train : ', sum(y_train_accuracy) / len(y_train_accuracy))
        # print('Average acc val : ', sum(y_valid_accuracy) / len(y_valid_accuracy))
        plt.savefig('graphs/acc_'+ self.model_type +'_bcm.png')

    def get_final_result_multi(self):

        final_result_multi = dict()
        prediction_funtion = NNMultiClassModel.predict_based_on_bert_classifier
        
        valid_2016_context_embs = self.data_embd['valid_2016']['context_embds']
        valid_2016_ending_0_embs = self.data_embd['valid_2016']['ending0_embds']
        valid_2016_ending_1_embs =  self.data_embd['valid_2016']['ending1_embds']
        valid_2016_data = self.dataset['valid_2016']
        prediction_2016 = prediction_funtion(self, valid_2016_context_embs, valid_2016_ending_0_embs, valid_2016_ending_1_embs)
        final_result_multi['valid_2016'] = compute_accuracy(valid_2016_data, prediction_2016)

        valid_2018_context_embs = self.data_embd['valid_2018']['context_embds']
        valid_2018_ending_0_embs = self.data_embd['valid_2018']['ending0_embds']
        valid_2018_ending_1_embs =  self.data_embd['valid_2018']['ending1_embds']
        valid_2018_data = self.dataset['valid_2018']
        prediction_2018 = prediction_funtion(self, valid_2018_context_embs, valid_2018_ending_0_embs, valid_2018_ending_1_embs)
        final_result_multi['valid_2018'] = compute_accuracy(valid_2018_data, prediction_2018)

        test_context_embs = self.data_embd['test']['context_embds']
        test_ending_0_embs = self.data_embd['test']['ending0_embds']
        test_ending_1_embs =  self.data_embd['test']['ending1_embds']
        test_data = self.dataset['test']
        prediction_test = prediction_funtion(self, test_context_embs, test_ending_0_embs, test_ending_1_embs)
        final_result_multi['test'] = compute_accuracy(test_data, prediction_test)

        return final_result_multi

    def get_final_result_binary(self):

        get_final_result_binary = dict()
        prediction_funtion = NNBinaryClassModel.predict_based_on_bert_binary_classifier

        test_context_embs = self.data_embd['test']['context_embds']
        test_ending_0_embs = self.data_embd['test']['ending0_embds']
        test_ending_1_embs =  self.data_embd['test']['ending1_embds']
        test_data = self.dataset['test']
        prediction_test = prediction_funtion(self, test_context_embs, test_ending_0_embs, test_ending_1_embs)
        get_final_result_binary['test'] = compute_accuracy(test_data, prediction_test)

        return get_final_result_binary

    def scores_lists_multi(self, context_embs, ending_0_embs, ending_1_embs):
        """Returns a list of predictions based on model."""
        predicted_embs = self.model(context_embs) # return array [total_dataset, 768]
        score_list = []
        label = []

        for idx in range(predicted_embs.shape[0]):
            pred_emb = predicted_embs[idx, :]
            score_0 = np.dot(pred_emb, ending_0_embs[idx, :])
            score_1 = np.dot(pred_emb, ending_1_embs[idx, :])
            score_list.append(softmax([score_0, score_1]))
            label.append(int(score_0 < score_1))

        return score_list, label

    def scores_list_binary(self, context_embs, ending_0_embs, ending_1_embs):
        scores_ending_0 = self.model(tf.concat([context_embs, ending_0_embs], -1))
        # print("scores_ending_0 : ", np.array(scores_ending_0))
        scores_ending_1 = self.model(tf.concat([context_embs, ending_1_embs], -1))
        # print("scores_ending_1 : ", np.array(scores_ending_1))
        score_list = []
        for score0, score1 in zip(scores_ending_0[:, 1], scores_ending_1[:, 1]):
            score_list.append(softmax([score0, score1]))
        # print("scores : ", scores)
        label = [int(x < y) for x,y in zip(scores_ending_0[:, 1], scores_ending_1[:, 1])]
        # print('chosen_label : ', chosen_label)
        return score_list, label
    
    def create_tables(self):

        if self.type == 'multi':
            scores, chosen_label = self.scores_lists_multi(self.data_embd['test']['context_embds'], self.data_embd['test']['ending0_embds'], self.data_embd['test']['ending1_embds'])
        elif self.type == 'binary':
            scores, chosen_label = self.scores_list_binary(self.data_embd['test']['context_embds'], self.data_embd['test']['ending0_embds'], self.data_embd['test']['ending1_embds'])

        ground_truth = [ex['label'] for ex in self.dataset['test']]

        verdict = [x == y for x,y in zip(chosen_label, ground_truth)]

        table = {'No': range(1, len(ground_truth)+1),
                'Probability Ending 1': [x[0] for x in scores],
                'Probability Ending 2': [x[1] for x in scores],
                'Ground Truth': [el+1 for el in ground_truth],
                'Verdict':verdict
                }
                
        df = pd.DataFrame(table, columns = ['No', 'Probability Ending 1', 'Probability Ending 2', 'Ground Truth', 'Verdict'])
        df = df.sort_values(by=['Verdict', 'Probability Ending 1'])
        df.to_csv('tables/table_'+ self.type +'_bert.csv', sep='\t')

        df_90 = df.loc[(df['Probability Ending 1'] > 0.9) | (df['Probability Ending 2'] > 0.9)] 
        df_90_true = df_90.loc[(df['Verdict'] == True)]
        df_80 = df.loc[(df['Probability Ending 1'] > 0.8) | (df['Probability Ending 2'] > 0.8)] 
        df_80_true = df_80.loc[(df['Verdict'] == True)]
        df_70 = df.loc[(df['Probability Ending 1'] > 0.7) | (df['Probability Ending 2'] > 0.7)] 
        df_70_true = df_70.loc[(df['Verdict'] == True)]
        df_60 = df.loc[(df['Probability Ending 1'] > 0.6) | (df['Probability Ending 2'] > 0.6)] 
        df_60_true = df_60.loc[(df['Verdict'] == True)]
        df_50 = df.loc[(df['Probability Ending 1'] > 0.5) | (df['Probability Ending 2'] > 0.5)] 
        df_50_true = df_50.loc[(df['Verdict'] == True)]

        scores =  [ df_50_true.shape[0]-df_60_true.shape[0],
                    df_60_true.shape[0]-df_70_true.shape[0],
                    df_70_true.shape[0]-df_80_true.shape[0],
                    df_80_true.shape[0]-df_90_true.shape[0],
                    df_90_true.shape[0]
                    ]

        return scores

class PlotScore():

    def __init__(self, scoreList):
        self.list_score = scoreList

    def create_bar_chart(self):
        pass

if __name__ == '__main__':

    from data import Data

    data = { 'train' : Data('data/train2017.csv').get_train_data(),
             'valid_2016' : Data('data/valid2016.csv').get_validtest_data(),
             'valid_2018' : Data('data/valid2018.csv').get_validtest_data(),
             'test' : Data('data/test2016.csv').get_validtest_data() }

    training_data_multi = { 'X_train_step' : list(range(10)), 
                                 'Y_VALUE' : {
                                    'train_loss' : list(range(10)),
                                    'valid_2016_loss' : list(range(10)),
                                    'valid_2018_loss' : list(range(10)),
                                    'train_accuracy' : list(range(10)),
                                    'valid_2016_accuracy' : list(range(10)),
                                    'valid_2018_accuracy' : list(range(10))}
                                }
    training_data_binary = { 'X_train_step' : list(range(10)),
                             'Y_VALUE' : {
                                'train_loss' : list(range(10)),
                                'valid_loss' : list(range(10)),
                                'train_accuracy' : list(range(10)),
                                'valid_accuracy' : list(range(10)),
                          }
                        }  
    import pickle
    import tensorflow as tf

    trained_model_multi = tf.keras.models.load_model('dummy/best_model_sbert.h5') 
    trained_model_binary = tf.keras.models.load_model('dummy/best_model_sbert_2.h5') 
    embedded_data = dict()


    with open('dummy/train_sentencebert-wk.pkl', 'rb') as f:
        load_data = pickle.load(f)
        embedded_data['train'] = {'context' : load_data['contexts'], 
                                  'ending' : load_data['endings'] }

    with open('dummy/valid_2016_sentencebert-wk.pkl', 'rb') as f:
        load_data = pickle.load(f)
        embedded_data['valid_2016'] = { 'context' : load_data['contexts'],
                                        'ending0' : load_data['endings_0'], 
                                        'ending1' : load_data['endings_1'] }

    with open('dummy/valid_2018_sentencebert-wk.pkl', 'rb') as f:
        load_data = pickle.load(f)
        embedded_data['valid_2018'] = { 'context' : load_data['contexts'],
                                        'ending0' : load_data['endings_0'], 
                                        'ending1' : load_data['endings_1'] }

    with open('dummy/test_2016_sentencebert-wk.pkl', 'rb') as f:
        load_data = pickle.load(f)
        embedded_data['test'] = { 'context' : load_data['contexts'],
                                  'ending0' : load_data['endings_0'], 
                                  'ending1' : load_data['endings_1'] }

    model_type = 'sbert'

    final_result_multi = Result(trained_model_multi, embedded_data, data, training_data_multi, 'multi', model_type).get_final_result_multi()
    Result(trained_model_multi, embedded_data, data, training_data_multi, 'multi', model_type).create_graph_multi()

    final_result_binary = Result(trained_model_binary, embedded_data, data, training_data_binary, 'binary', model_type).get_final_result_binary()    
    Result(trained_model_binary, embedded_data, data, training_data_binary, 'binary', model_type).create_graph_binary()

    score_data_multi = Result(trained_model_multi, embedded_data, data, training_data_multi, 'multi', model_type).create_tables()
    score_data_binary = Result(trained_model_binary, embedded_data, data, training_data_binary, 'binary', model_type).create_tables()

    print(score_data_multi, score_data_binary)
