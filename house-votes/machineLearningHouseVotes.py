# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np

# https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
# Load dataset
url = "house-votes-84.data"

dataset = pandas.read_csv(url, sep=',')

# instances (rows) and attributes (columns)
print(dataset.shape)
print('\n\n')

# peek at dataset
print(dataset.head(20))
print('\n\n')

# descriptions
print(dataset.describe())
print('\n\n')

# class distribution
print(dataset.groupby('party').size())
print('\n\n')

# dict of label encoders
labelEncoders = {}


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            le = preprocessing.LabelEncoder()           
            column_contents = df[column].values.tolist()
            le.fit(column_contents)
            df[column] = le.transform(column_contents)
            
            # each column gets own label encoder
            labelEncoders[column] = le
    return df


# helps ensure classification report has correct labels
def get_report_labels():
    labelEncoder = labelEncoders['party']
    label_count = len(labelEncoder.classes_)
    all_labels = labelEncoder.inverse_transform(np.arange(label_count))

    predicted_labels = []

    distinct_values = pandas.unique(pandas.Series(predictions))
    for distinct_value in distinct_values:
        predicted_labels.append(all_labels[distinct_value])

    return predicted_labels    
    
    

dataset = handle_non_numerical_data(dataset)

spot_check_record = 39
# strip single instance from dataset for spot checking prediction later
single_instance = dataset.loc[[spot_check_record]]
# drop index 
dataset.drop([spot_check_record], inplace=True)




# single instance features
single_instance_features = single_instance.values[:,1:]
# single instance classification
single_instance_class = single_instance.values[:,0]




# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False)
plt.show()

# histograms
#####dataset.hist()
#####plt.show()

# scatter plot matrix
####scatter_matrix(dataset)
####plt.show()

# Split-out validation dataset
array = dataset.values

# features
X = array[:,1:]

# classifications
Y = array[:,0]

validation_size = 0.20
seed = 7

# holdout cross validation
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
##models.append(('MLPC', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    # k fold cross validation
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
print('\n\n')
	
# Compare Algorithms
"""
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)"""
#######plt.show()

# Make predictions on validation dataset w model of your choice
model = LinearDiscriminantAnalysis()
print('using model: ' + str(model))

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('\naccuracy score:')
print(accuracy_score(Y_validation, predictions))
print('\nconfusion matrix:')
print(confusion_matrix(Y_validation, predictions))


report_labels = get_report_labels()

print('\nclassification report:')
print(classification_report(Y_validation, predictions, target_names=report_labels))

print("\npredictions:")
print(predictions)

print("\nY_validation:")
print(Y_validation)

labelEncoder = labelEncoders['party']

print("\nspot check record: " + str(spot_check_record))
print("spot check single instance prediction:")
print(labelEncoder.inverse_transform(model.predict(single_instance_features)))


print("spot check single instance answer:")
print(labelEncoder.inverse_transform(single_instance_class))





