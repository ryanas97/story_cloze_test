import pickle

def save_embds(embdDict, typeData, typeModel):
    if typeData == 'train':
        mydict = {'context_embds': embdDict['context_embds'], 'ending_embds': embdDict['ending_embds']}
        output = open('embds/' + typeData + '_' + typeModel + '.pkl', 'wb')
        pickle.dump(mydict, output)
        output.close()
    elif typeData == 'valid_2016' or typeData == 'valid_2018' or typeData == 'test':
        mydict = {'context_embds': embdDict['context_embds'], 'ending0_embds': embdDict['ending0_embds'], 'ending1_embds': embdDict['ending1_embds']}
        output = open('embds/' + typeData + '_' + typeModel + '.pkl', 'wb')
        pickle.dump(mydict, output)
        output.close()

def load_embds(typeData, typeModel):
    if typeData == 'train':
        with open('embds/' + typeData + '_' + typeModel + '.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['context_embds'], data['ending_embds']

    elif typeData == 'valid_2016' or typeData == 'valid_2018' or typeData == 'test':
        with open('embds/' + typeData + '_' + typeModel + '.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['context_embds'], data['ending0_embds'], data['ending1_embds']

def save_training_data(trainingData, typeNN, typeModel):
    mydict = dict()

    if typeNN == 'multi':
        mydict = { 'X_train_step' : trainingData['X_train_step'], 'Y_VALUE' : dict() }
        mydict['Y_VALUE'] = {'train_loss' : trainingData['Y_VALUE']['train_loss'], 
                             'valid_2016_loss' : trainingData['Y_VALUE']['valid_2016_loss'],
                             'valid_2018_loss' : trainingData['Y_VALUE']['valid_2018_loss'],
                             'train_accuracy' : trainingData['Y_VALUE']['train_accuracy'],
                             'valid_2016_accuracy' : trainingData['Y_VALUE']['valid_2016_accuracy'],
                             'valid_2018_accuracy' : trainingData['Y_VALUE']['valid_2018_accuracy'] }

        output = open('training_data/' + typeNN + '_' + typeModel + '.pkl', 'wb')
        pickle.dump(mydict, output)
        output.close()

    elif typeNN == 'binary':
        mydict = { 'X_train_step' : trainingData['X_train_step'], 'Y_VALUE' : dict() }
        mydict['Y_VALUE'] = {'train_loss' : trainingData['Y_VALUE']['train_loss'], 
                             'valid_loss' : trainingData['Y_VALUE']['valid_loss'],
                             'train_accuracy' : trainingData['Y_VALUE']['train_accuracy'],
                             'valid_accuracy' : trainingData['Y_VALUE']['valid_accuracy'] }
        
        output = open('training_data/' + typeNN + '_' + typeModel + '.pkl', 'wb')
        pickle.dump(mydict, output)
        output.close()

def load_training_data(typeNN, typeModel):
    with open('training_data/' + typeNN + '_' + typeModel + '.pkl', 'rb') as f:
        data = pickle.load(f)

    return data
