from data import Data
from models import MyBERTModel, NNMultiClassModel, NNBinaryClassModel
from evals import Result, PlotScore
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModel, AutoConfig
from saveload import save_embds, load_embds, load_training_data
from os import path
import tensorflow as tf

def main():
    # Data Retrieval for training, validation and test data.
    csv_data = {'train' : Data('data/train2017.csv').get_train_data(),
                'valid_2016' : Data('data/valid2016.csv').get_validtest_data(),
                'valid_2018' : Data('data/valid2018.csv').get_validtest_data(),
                'test' : Data('data/test2016.csv').get_validtest_data() }

    # Transformers has a unified API
    # for 10 transformer architectures and 30 pretrained weights.
    #          Model          | Tokenizer         | Config          | Pretrained weights shortcut  | Type
    MODELS = [(BertModel,     BertTokenizer,       ''           ,   'bert-base-uncased',           'bert'),
              (RobertaModel,  RobertaTokenizer,       ''           ,   'roberta-base',               'roberta')
            # (AutoModel,       AutoTokenizer,       AutoConfig   ,   'deepset/sentence_bert',       'sbert')
            ]

    for model_class, tokenizer_class, config_class, pretrained_weights, model_type in MODELS:
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # if config_class != '': 
        #     config = config_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        #     model = model_class.from_pretrained(pretrained_weights, from_tf=True, config=config)
        # else:
        model = model_class.from_pretrained(pretrained_weights)

        embedded_data = dict()
        embedded_data['train'], embedded_data['valid_2016'], embedded_data['valid_2018'], embedded_data['test'] = [dict() for _ in range(4)]

        # Create embedded vectors
        for data_type in ['train', 'valid_2016', 'valid_2018', 'test']:
            if data_type == 'train':
                if path.exists('embds/' + data_type + '_' + model_type + '.pkl'):
                    embedded_data[data_type]['context_embds'], embedded_data[data_type]['ending_embds'] = load_embds(data_type, model_type)
                else:
                    embedded_data[data_type]['context_embds'], embedded_data[data_type]['ending_embds'] = MyBERTModel(csv_data[data_type], model, tokenizer).get_train_embeddings()
                    save_embds(embedded_data[data_type], data_type, model_type)

            elif data_type == 'valid_2016' or data_type == 'valid_2018' or data_type == 'test':
                if path.exists('embds/' + data_type + '_' + model_type + '.pkl'):
                    embedded_data[data_type]['context_embds'], embedded_data[data_type]['ending0_embds'], embedded_data[data_type]['ending1_embds'] = load_embds(data_type, model_type)
                else:
                    embedded_data[data_type]['context_embds'], embedded_data[data_type]['ending0_embds'], embedded_data[data_type]['ending1_embds'] = MyBERTModel(csv_data[data_type], model, tokenizer).get_validtest_embeddings()
                    save_embds(embedded_data[data_type], data_type, model_type)

        # Feed the embedded vectors to the neural network models.
        trained_model, training_data = [dict() for _ in range(2)]

        if path.exists('final_models/best_model_' + 'multi' + '_' + model_type + '.h5') and path.exists('training_data/' + 'multi' + '_' + model_type + '.pkl'):
            trained_model['multi'], training_data['multi'] = tf.keras.models.load_model('final_models/best_model_' + 'multi' + '_' + model_type + '.h5'), load_training_data('multi', model_type) 
        else:
            trained_model['multi'], training_data['multi'] = NNMultiClassModel(embedded_data, csv_data, model_type).train_model()
            
        if path.exists('final_models/best_model_' + 'binary' + '_' + model_type + '.h5') and path.exists('training_data/' + 'binary' + '_' + model_type + '.pkl'):
            trained_model['binary'], training_data['binary'] = tf.keras.models.load_model('final_models/best_model_' + 'binary' + '_' + model_type + '.h5'), load_training_data('binary', model_type) 
        else:
            trained_model['binary'], training_data['binary'] = NNBinaryClassModel(embedded_data, csv_data, model_type).train_model()

        # Get final prediction and create a graph file
        final_result = dict()

        final_result['multi'] = Result(trained_model['multi'], embedded_data, csv_data, trained_model['multi'], 'multi', model_type).get_final_result_multi()
        Result(trained_model['multi'], embedded_data, csv_data, training_data['multi'], 'multi', model_type).create_graph_multi()

        final_result['binary'] = Result(trained_model['binary'], embedded_data, csv_data, trained_model['binary'], 'binary', model_type).get_final_result_binary()    
        Result(trained_model['binary'], embedded_data, csv_data, training_data['binary'], 'binary', model_type).create_graph_binary()

        print('The '+ model_type.upper() + ' + NN Multi-Class Model capable of achieveing accuracy of ', round(final_result['multi']['test'] * 100, 2), '% on the Story Cloze Test')
        print('The '+ model_type.upper() + ' + NN Binary-Class Model capable of achieveing accuracy of ', round(final_result['binary']['test'] * 100, 2), '% on the Story Cloze Test')

        score_data_multi = Result(trained_model['multi'], embedded_data, csv_data, training_data['multi'], 'multi', model_type).create_tables()
        score_data_binary = Result(trained_model['binary'], embedded_data, csv_data, training_data['binary'], 'binary', model_type).create_tables()

        print('Score Range 50-60-70-80-90 for Multi and ' + model_type.title() + ' model : ', score_data_multi)
        print('Score Range 50-60-70-80-90 for Binary and ' + model_type.title() + ' model : ', score_data_binary)

if __name__ == '__main__':
    main()