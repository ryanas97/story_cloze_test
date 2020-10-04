import csv

class Data:

    def __init__(self, path):
        self.path = path

    def get_train_data(self):
        complete_data = []
        with open(self.path, encoding='cp932', errors='ignore') as f:
            reader = csv.DictReader(f)
            for line in reader:
                story = [line['sentence1'], line['sentence2'],
                        line['sentence3'], line['sentence4'],
                        line['sentence5']]
                complete_data.append({'story': story})
        return complete_data

    def get_validtest_data(self):
        complete_data = []
        with open(self.path, encoding='cp932', errors='ignore') as f:
            reader = csv.DictReader(f)
            for line in reader:
                context = [line['InputSentence1'], line['InputSentence2'],
                        line['InputSentence3'], line['InputSentence4']]
                option_0 = line['RandomFifthSentenceQuiz1']
                option_1 = line['RandomFifthSentenceQuiz2']
                label = int(line['AnswerRightEnding']) - 1
                complete_data.append({'context': context, 
                                'options': [option_0, option_1],
                                'label': label})
        return complete_data

if __name__ == '__main__':

    from data import Data
    train_data = Data('data/train2017.csv').get_train_data()
    print(train_data[-1], len(train_data))
    valid_2016_data = Data('data/valid2016.csv').get_validtest_data()
    valid_2018_data = Data('data/valid2018.csv').get_validtest_data()
    test_data = Data('data/test2016.csv').get_validtest_data()
    print(valid_2016_data[-1], len(valid_2016_data))
    print(valid_2018_data[-1], len(valid_2018_data))
    print(test_data[-1], len(test_data))



