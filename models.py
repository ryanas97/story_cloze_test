from data import Data
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import tensorflow as tf
import datetime
import random
from saveload import save_training_data

class MyBERTModel:

    def __init__(self, data, model, tokenizer):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer

    def bertEmbedding(self, data):
        """Load the data to the pre-trined BERT Model."""
        inputs_ids = self.tokenizer.encode(data) #Convert text to a list of a sequence of ids (integer), using the tokenizer and vocabulary.
        input_ids = torch.tensor(inputs_ids).unsqueeze(0)  # Batch size 1

        _, merged_embedding = self.model(input_ids)
        return merged_embedding.detach().numpy()

    def get_train_embeddings(self):
        """Computes embeddings for each example in the provided train set."""
        context_embeddings = []
        ending_embeddings = []
        print('Starting')

        for idx, example in enumerate(self.data):

            if idx % 100 == 0:
                print('{}/{}'.format(idx+1, len(self.data)))
                print(' '.join(example['story']))

            context_embedding = self.bertEmbedding(' '.join(example['story'][:4])) #Embeeding of first 4 sentences (dim 1 x 768)
            ending_embedding = self.bertEmbedding(example['story'][4]) #Embeeding of 5th sentence (dim 1 x 768)

            context_embeddings.append(context_embedding) 
            ending_embeddings.append(ending_embedding)
        context_embeddings = np.concatenate(context_embeddings, axis=0) #Concatenate Embbeding (dim n x 768)
        ending_embeddings = np.concatenate(ending_embeddings, axis=0) #Concatenate Embbeding (dim n x 768)
        return context_embeddings, ending_embeddings

    def get_validtest_embeddings(self):
        """Computes embeddings for each example in the provided validation set."""
        context_embeddings = []
        ending_0_embeddings = []
        ending_1_embeddings = []
        for idx, example in enumerate(self.data):

            if idx % 100 == 0:
                print('{}/{}'.format(idx+1, len(self.data)))
                print(' '.join(example['context'][:4]), '\n', example['options'][0], example['options'][1])

            context_embedding = self.bertEmbedding(' '.join(example['context'][:4])) #Embeeding of first 4 sentences (dim 1 x 768)
            ending_0_embedding = self.bertEmbedding(example['options'][0]) #Embeeding of first options (dim 1 x 768)
            ending_1_embedding = self.bertEmbedding(example['options'][1]) #Embeeding of second options (dim 1 x 768)

            context_embeddings.append(context_embedding)
            ending_0_embeddings.append(ending_0_embedding)
            ending_1_embeddings.append(ending_1_embedding)

        context_embeddings = np.concatenate(context_embeddings, axis=0) #Concatenate Embbeding (dim n x 768)
        ending_0_embeddings = np.concatenate(ending_0_embeddings, axis=0) #Concatenate Embbeding (dim n x 768)
        ending_1_embeddings = np.concatenate(ending_1_embeddings, axis=0) #Concatenate Embbeding (dim n x 768)
        return context_embeddings, ending_0_embeddings, ending_1_embeddings

class NNMultiClassModel:

    def __init__(self, embdData, dataset, typeModel):
        self.train_embd = embdData['train']
        self.valid_2016_embd = embdData['valid_2016']
        self.valid_2018_embd = embdData['valid_2018']
        self.dataset = dataset
        self.model_type = typeModel
        #### HYPERPARAMETERS ####
        self.hyperparameters = { 'NUM_TRAIN_STEPS' : 10000, # How many step to train for.
                                 'BATCH_SIZE' : 64, # Number of examples used in step of training.
                                 'NUM_CANDIDATES' : 100, # Number of candidate 5th sentences classifier must decide between.
                                 'LEARNING_RATE' : 0.0001 } # Learning rate.
        # If loss is barely going down, learning rate might be too small.
        # If loss is jumping around, it might be too big.
        self.model = self.get_model()
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.hyperparameters['LEARNING_RATE'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def get_model(self):
        """Returns a Keras model.
        The model should input a [batch_size, embedding_size] tensor and output a new
        [batch_size, embedding_size] tensor. At it's simplest, it could just be a
        single dense layer. 
        """

        # This is an example of a very simple network consisting of a single nonlinear
        # layer followed by a linear projection back to the BERT embedding size.
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, input_shape=(768,) , activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.05)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1024, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.05)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(768, activation="linear"))

        return model

    def get_batch(self):
        """Returns a single training batch.
        
        Returns:
        batch_inputs: [batch_size, embedding_size] matrix of context embeddings.
        batch_candidates: [num_candidates, embedding_size] matrix of embeddings of 
            candidate 5th sentence embeddings. The groundtruth 5th sentence for the ith
            example in batch_inputs is in the ith row of batch_candidates.
        labels: [batch_size] For each example in batch_inputs, the index of the true
            5th sentence in batch_candidates.
        """

        if self.hyperparameters['NUM_CANDIDATES'] < self.hyperparameters['BATCH_SIZE']:
            raise ValueError(
                'At minimum the number of candidates is at least all of the other 5th '
                'sentences in the batch.')
            
        batch_inputs = []
        batch_candidates = []
        batch_labels = []

        list_index = random.sample(range(0, self.train_embd['context_embds'].shape[0]-1), self.hyperparameters['BATCH_SIZE'])

        for i, index in enumerate(list_index):
            batch_inputs.append(self.train_embd['context_embds'][index, :]) 
            batch_candidates.append(self.train_embd['ending_embds'][index, :])
            # The true next embedding is in the ith position in the candidates
            batch_labels.append(i)

        # Increase the number of "distractor" candidates to num_candidates by (num_candidates - batch_size).
        for i in range(self.hyperparameters['NUM_CANDIDATES'] - self.hyperparameters['BATCH_SIZE']):
            rand_ex_index = random.randint(0, self.train_embd['context_embds'].shape[0]-1)
            batch_candidates.append(self.train_embd['ending_embds'][rand_ex_index, :])

        batch_inputs = np.stack(batch_inputs, axis=0) # return array [BATCH_SIZE, 768]
        batch_candidates = np.stack(batch_candidates, axis=0) # return array [NUM_CANDIDATES, 768]

        return batch_inputs, batch_candidates, batch_labels

    def loss_function_validation(self, context_embs, ending_0_embs, ending_1_embs, validData):
        """Returns the value of loss with loss function."""

        predicted_embs = self.model(context_embs) # return array [total_dataset, 768]
        logits = []

        for i,v in enumerate(np.swapaxes([ending_0_embs, ending_1_embs],0,1)):
            logits.append(tf.squeeze(tf.matmul([predicted_embs[i]], v, transpose_b=True)))
        
        logits = tf.convert_to_tensor(logits)
        loss_value = self.loss_fn(np.array([ex['label'] for ex in validData]), logits)
        return loss_value

    def predict_based_on_bert_classifier(self, context_embs, ending_0_embs, ending_1_embs):
        """Returns a list of predictions based on model."""
        predicted_embs = self.model(context_embs) # return array [total_dataset, 768]

        predictions = []
        for idx in range(predicted_embs.shape[0]):
            pred_emb = predicted_embs[idx, :]
            score_0 = np.dot(pred_emb, ending_0_embs[idx, :])
            score_1 = np.dot(pred_emb, ending_1_embs[idx, :])
            predictions.append(score_0 < score_1)
        return predictions

    def train_model(self):
        
        data_training = { 'X_train_step' : [],
                          'Y_VALUE' : {
                              'train_loss' : [],
                              'valid_2016_loss' : [],
                              'valid_2018_loss' : [],
                              'train_accuracy' : [],
                              'valid_2016_accuracy' : [],
                              'valid_2018_accuracy' : []}
                          }
        min_loss = 1000

        print('\n====================== TRAINING Model ' + self.model_type.title() + ' + Multi STARTED ======================')

        for train_step in range(self.hyperparameters['NUM_TRAIN_STEPS']+1):
    
            with tf.GradientTape() as tape:
                batch_inputs, batch_candidates, batch_labels = self.get_batch()

                # Predicted 5th sentence embedding for each batch position/
                outputs = self.model(batch_inputs) # keras model. return list [BATCH_SIZE, 768]
                # The logits will be batch_size * num_candidates, giving a score for each
                # candidate 5th sentence. We'd like the true 5th sentence to have the highest score.
                logits = tf.matmul(outputs, batch_candidates, transpose_b=True) # array [outputs, batch_candidates]
                # Loss value for this minibatch
                loss_value = self.loss_fn(batch_labels, logits)

            if train_step % 100 == 0:
                train_acc = []
                pred_truth = []

                for i,v in enumerate(logits):
                    index_max_top5 = np.array(v).argsort()[-10:][::-1]
                    pred_truth.append((index_max_top5, i))
                    train_acc.append(i in index_max_top5)

                print('Step {}, batch_train_loss={:.3f}'.format(train_step, loss_value),
                      'train accuracy : {}/{} = {:.3f}'.format(np.sum(train_acc), len(train_acc), np.sum(train_acc)/len(train_acc)) 
                    )
                loss_value_record_2016 = self.loss_function_validation(self.valid_2016_embd['context_embds'], self.valid_2016_embd['ending0_embds'], self.valid_2016_embd['ending1_embds'], self.dataset['valid_2016'])
                loss_value_record_2018 = self.loss_function_validation(self.valid_2018_embd['context_embds'], self.valid_2018_embd['ending0_embds'], self.valid_2018_embd['ending1_embds'], self.dataset['valid_2018'])

                if min_loss > (loss_value_record_2016 + loss_value_record_2018)/2:
                    min_loss = (loss_value_record_2016 + loss_value_record_2018)/2
                    self.model.save('final_models/best_model_multi_'+ self.model_type + '.h5')
                    print("Best model checkpoint for step {0} with average loss of : {1:.3f} \n".format(
                    int(train_step), (loss_value_record_2016 + loss_value_record_2018)/2))

            if train_step % 500 == 0:
                loss_value_2016 = self.loss_function_validation(self.valid_2016_embd['context_embds'], self.valid_2016_embd['ending0_embds'], self.valid_2016_embd['ending1_embds'], self.dataset['valid_2016'])
                predictions_2016 = self.predict_based_on_bert_classifier(self.valid_2016_embd['context_embds'], self.valid_2016_embd['ending0_embds'], self.valid_2016_embd['ending1_embds'])
                loss_value_2018 = self.loss_function_validation(self.valid_2018_embd['context_embds'], self.valid_2018_embd['ending0_embds'], self.valid_2018_embd['ending1_embds'], self.dataset['valid_2018'])
                predictions_2018 = self.predict_based_on_bert_classifier(self.valid_2018_embd['context_embds'], self.valid_2018_embd['ending0_embds'], self.valid_2018_embd['ending1_embds'])

                print('2016 validation loss: {:.3f}'.format(loss_value_2016), 
                    'validation accuracy: {}/{} = {:.3f}'.format(
                        np.sum(np.equal(np.array([ex['label'] for ex in self.dataset['valid_2016']]), np.array(predictions_2016))),
                        len(np.array([ex['label'] for ex in self.dataset['valid_2016']])), 
                        compute_accuracy(self.dataset['valid_2016'], predictions_2016)))
                print('2018 validation loss: {:.3f}'.format(loss_value_2018),
                    'validation accuracy: {}/{} = {:.3f}'.format(
                        np.sum(np.equal(np.array([ex['label'] for ex in self.dataset['valid_2018']]), np.array(predictions_2018))),
                        len(np.array([ex['label'] for ex in self.dataset['valid_2018']])),
                        compute_accuracy(self.dataset['valid_2018'], predictions_2018)))
            
                data_training['X_train_step'].append(train_step)
                data_training['Y_VALUE']['train_loss'].append(loss_value)
                data_training['Y_VALUE']['valid_2016_loss'].append(loss_value_2016)
                data_training['Y_VALUE']['valid_2018_loss'].append(loss_value_2018)
                data_training['Y_VALUE']['train_accuracy'].append(np.sum(train_acc)/len(train_acc))
                data_training['Y_VALUE']['valid_2016_accuracy'].append(compute_accuracy(self.dataset['valid_2016'], predictions_2016))
                data_training['Y_VALUE']['valid_2018_accuracy'].append(compute_accuracy(self.dataset['valid_2018'], predictions_2018))


            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        save_training_data(data_training, 'multi', self.model_type)
        return self.model, data_training

class NNBinaryClassModel:

    def __init__(self, embdData, dataset, typeModel):
        self.valid_2016_embd = embdData['valid_2016']
        self.valid_2018_embd = embdData['valid_2018']
        self.dataset = dataset
        self.model_type = typeModel
        #### HYPERPARAMETERS ####
        self.hyperparameters = { 'NUM_TRAIN_STEPS' : 10000, # How many step to train for.
                                 'BATCH_SIZE' : 64, # Number of examples used in step of training.
                                 'LEARNING_RATE' : 0.0001, # Learning rate.
                                 'NUM_TRAIN_EXAMPLES' : 5000 } # How many examples from the valid set to use for training.
                                                               # The remainder will be placed into a new valid set.
        # You should with varying NUM_TRAIN_EXAMPLES. If it is larger, you will train a 
        # better model, but you will have fewer examples available your validation set
        # for tuning other hyperparameters.
        # You may experiment with other optimizers or loss functions if you'd like.
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.hyperparameters['LEARNING_RATE'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = self.get_binary_classifier()


        self.all_inputs, self.all_labels = self.build_dataset()
        self.train_inputs = self.all_inputs[:self.hyperparameters['NUM_TRAIN_EXAMPLES'], :]
        self.train_labels = self.all_labels[:self.hyperparameters['NUM_TRAIN_EXAMPLES']]
        self.valid_inputs = self.all_inputs[self.hyperparameters['NUM_TRAIN_EXAMPLES']:, :]
        self.valid_labels = self.all_labels[self.hyperparameters['NUM_TRAIN_EXAMPLES']:]

    def predict_based_on_bert_binary_classifier(self, context_embs, ending_0_embs, ending_1_embs):
        """Returns a list of predictions based on binary classification model."""
        scores_ending_0 = self.model(tf.concat([context_embs, ending_0_embs], -1))
        scores_ending_1 = self.model(tf.concat([context_embs, ending_1_embs], -1))
        predictions = tf.less(scores_ending_0, scores_ending_1)[:, 1]
        return predictions
    
    def get_binary_classifier(self):
        """Returns a Keras model.
        The model should input a [batch_size, 2*embedding_size] tensor and output a
        [batch_size, 2] tensor. The final final dimension needs to be 2 because we are
        doing binary classification. """

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, input_shape=(1536,), activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.05)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(2, activation="linear"))
        
        return model

    def build_dataset(self):
        """Builds a dataset out of the validation set examples.

        Each example in valid_2016 and valid_2018 becomes two exampes in this new 
        dataset:
        * one where ending_0's embedding is concatenated to the context embedding
        * one where ending_1's embedding is concatenated to the context embedding

        The label for each example is 1 if the correct ending's embedding is present,
        0 if the incorrect ending's embedding is present.

        Returns:
        all_inputs: [new_dataset_size, embedding_size*2]
        all_labels: [new_dataset_size]
        """
        inputs_2016 = tf.concat(
            [tf.concat([self.valid_2016_embd['context_embds'], self.valid_2016_embd['ending0_embds']], axis=-1),         # C E0       
            tf.concat([self.valid_2016_embd['context_embds'], self.valid_2016_embd['ending1_embds']], axis=-1)], axis=0) # C E1
        labels = [int(ex['label']  == 0) for ex in self.dataset['valid_2016']]
        labels_2016 = labels + [1 - label for label in labels] # RA ~RA

        inputs_2018 = tf.concat(
            [tf.concat([self.valid_2018_embd['context_embds'], self.valid_2018_embd['ending0_embds']], axis=-1),
            tf.concat([self.valid_2018_embd['context_embds'], self.valid_2018_embd['ending1_embds']], axis=-1)], axis=0)
        labels = [int(ex['label']  == 0) for ex in self.dataset['valid_2018']]
        labels_2018 = labels + [1 - label for label in labels]

        all_inputs = tf.concat([inputs_2016, inputs_2018], axis=0)  # C16 E0-16
                                                                    # C16 E1-16
                                                                    # C18 E0-18
                                                                    # C18 E1-18
        all_labels = labels_2016 + labels_2018  # RA16 ~RA16 RA18 ~RA18
        return all_inputs, all_labels

        
    def get_batch_from_valid(self, batch_size, inputs, labels):
        """Returns a single training batch extracted form the validation set.

        Inputs:
        batch_size: The batch size.
        inputs: [dataset_size, 2*embedding_size] matrix of all inputs in the training
            set.
        labels: [dataset_size] for each example, 0 if example has the incorrect ending
            embedding, 1 if it has the correct ending embedding.
        
        Returns:
        batch_inputs: [batch_size, 2*embedding_size] matrix of embeddings (each
            embedding is a context embedding concatenated with an ending embedding).
        labels: [batch_size] For each example in batch_inputs, contains either 0 or 1,
            indicating whether the 5th ending is the correct one.
        """
        batch_inputs = []
        batch_labels = []
        for _ in range(batch_size):
            rand_ex_index = random.randint(0, inputs.shape[0]-1)    
            batch_inputs.append(inputs[rand_ex_index, :])
            batch_labels.append(labels[rand_ex_index])
            
        batch_inputs = np.stack(batch_inputs, axis=0)
        return batch_inputs, batch_labels

    def train_model(self):
        
        data_training = { 'X_train_step' : [],
                          'Y_VALUE' : {
                              'train_loss' : [],
                              'valid_loss' : [],
                              'train_accuracy' : [],
                              'valid_accuracy' : [],
                          }
                        }   
        min_loss = float('+inf')

        # Iterate over the batches of a dataset.
        print('\n====================== TRAINING Model ' + self.model_type.title() + ' + Binary STARTED ======================')

        for train_step in range(self.hyperparameters['NUM_TRAIN_STEPS']+1):
            with tf.GradientTape() as tape:
                batch_inputs, batch_labels = self.get_batch_from_valid(self.hyperparameters['BATCH_SIZE'], self.train_inputs, self.train_labels)
                logits = self.model(batch_inputs) 
                loss_value = self.loss_fn(batch_labels, logits)
                
            batch_acc = sum(tf.equal(batch_labels, tf.argmax(logits, axis=-1)).numpy()) / self.hyperparameters['BATCH_SIZE']
            valid_logits = self.model(self.valid_inputs)
            num_correct = sum(tf.equal(self.valid_labels, tf.argmax(valid_logits, axis=-1)).numpy())

            if train_step % 100 == 0:
                batch_acc = sum(tf.equal(batch_labels, tf.argmax(logits, axis=-1)).numpy()) / self.hyperparameters['BATCH_SIZE']
                print('Step {0}, batch_loss={1:.5f}, batch_acc : {2}/{3}={4:.3f}'.format(
                    train_step, loss_value, 
                    sum(tf.equal(batch_labels, tf.argmax(logits, axis=-1)).numpy()), self.hyperparameters['BATCH_SIZE'],
                    batch_acc))

            if train_step % 500 == 0:
                valid_logits = self.model(self.valid_inputs)
                loss_value_val = self.loss_fn(self.valid_labels, valid_logits)
                num_correct = sum(tf.equal(self.valid_labels, tf.argmax(valid_logits, axis=-1)).numpy())
                print('Validation loss : {0:.3f}, Validation accuracy: {1}/{2} = {3:.3f} '.format(
                    loss_value_val, num_correct, len(self.valid_labels), num_correct / len(self.valid_labels)))
                    
                if min_loss > loss_value_val:
                    min_loss = loss_value_val
                    self.model.save('final_models/best_model_binary_'+ self.model_type + '.h5')
                    print("Best model checkpoint for step {0} with average loss of : {1:.3f} \n".format(
                        int(train_step), loss_value_val))

                data_training['X_train_step'].append(train_step)
                data_training['Y_VALUE']['train_loss'].append(loss_value)
                data_training['Y_VALUE']['valid_loss'].append(loss_value_val)
                data_training['Y_VALUE']['train_accuracy'].append(batch_acc)
                data_training['Y_VALUE']['valid_accuracy'].append(num_correct / len(self.valid_labels))

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        save_training_data(data_training, 'binary', self.model_type)    

        return self.model, data_training

def compute_accuracy(data, predictions):
    ground_truth = np.array([ex['label'] for ex in data]) #Array of Label Dataset
    predictions = np.array(predictions)
    assert len(ground_truth) == len(predictions)

    return np.sum(np.equal(ground_truth, predictions)) / float(len(ground_truth))

if __name__ == '__main__':

    data = { 'train' : Data('data/train2017.csv').get_train_data(),
            'valid_2016' : Data('data/valid2016.csv').get_validtest_data(),
            'valid_2018' : Data('data/valid2018.csv').get_validtest_data(),
            'test' : Data('data/test2016.csv').get_validtest_data() }

    embedded_data = dict()
    train_context_embs, train_ending_embs = np.random.rand(5000,768), np.random.rand(5000,768)
    embedded_data['train'] = {'context' : train_context_embs, 'ending' : train_ending_embs}
    valid_2016_context_embs, valid_2016_ending_0_embs, valid_2016_ending_1_embs = np.random.rand(1871,768), np.random.rand(1871,768), np.random.rand(1871,768)
    valid_2018_context_embs, valid_2018_ending_0_embs, valid_2018_ending_1_embs = np.random.rand(1571,768), np.random.rand(1571,768), np.random.rand(1571,768)
    embedded_data['valid_2016'] = {'context' : valid_2016_context_embs, 'ending0' : valid_2016_ending_0_embs, 'ending1' : valid_2016_ending_1_embs}
    embedded_data['valid_2018'] = {'context' : valid_2018_context_embs, 'ending0' : valid_2018_ending_0_embs, 'ending1' : valid_2018_ending_1_embs}

    model_type = 'bert'

    trained_model_muticlass, training_data_multiclass = NNMultiClassModel(embedded_data, data, model_type).train_model()
    print(trained_model_muticlass, training_data_multiclass)
    trained_model_binary, training_data_binary = NNBinaryClassModel(embedded_data, data, model_type).train_model()
    print(trained_model_binary, training_data_binary)
