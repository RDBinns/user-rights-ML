from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier

tosTrainer = Trainer(tokenizer)

def get_corp(read_file):
	with open(read_file,"r") as r:
		corpus = []
		for line in r:
			tabsep = line.decode('utf-8').strip().split('\t')
			a = {}
			a['text'] = tabsep[0]
			a['rating'] = tabsep[1]
			corpus.append(a)
		return corpus

# get the corpus from a training set - using copyright clauses here as an example (a subset of the csv generated by the getpointsdata.py script)
tosSet = get_corp("tosdr.org/copyrighttrainset.txt")

# You need to train the system passing each text one by one to the trainer module.
for corpi in tosSet:
    tosTrainer.train(corpi['text'], corpi['rating'])

# When you have sufficient trained data, you are almost done and can start to use a classifier.
tosClassifier = Classifier(tosTrainer.data, tokenizer)

# Now you have a classifier which can give a try to classifiy text of policy clauses whose rating is unknown, yet. Example here drawn from test set
unknownInstance = "You are free to choose your own copyright license for your content in your account settings: Public Domain Creative Commons non commercial or free licenses but also classic copyright if you wish so."
classification = tosClassifier.classify(unknownInstance)

# the classification variable holds the possible categories sorted by their probablity value
print classification