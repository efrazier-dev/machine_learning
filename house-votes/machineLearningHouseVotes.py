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
import numpy as np

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
print(dataset.groupby('Class Name').size())
print('\n\n')

# TODO use label encoder instead??
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
            
            if column == 'Class Name':
                print('party mapping:')
                print(text_digit_vals)
                print('\n')
    return df



dataset = handle_non_numerical_data(dataset)

spot_check_record = 9
# strip single instance from dataset for spot checking prediction later
single_instance = dataset.loc[[spot_check_record]]
# drop index 
dataset.drop([spot_check_record], inplace=True)




# single instance features
single_instance_features = single_instance.values[:,1:]
# single instance classification
single_instance_class = single_instance.values[:,0]




# box and whisker plots
####dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
####plt.show()

# histograms
dataset.hist()
plt.show()

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
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


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
print("\nLogisticRegression:")
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print("\npredictions:")
print(predictions)

print("\nY_validation:")
print(Y_validation)

print("\nspot check record: " + str(spot_check_record))
print("spot check single instance prediction:")
print(model.predict(single_instance_features))

print("spot check single instance answer:")
print(single_instance_class)






