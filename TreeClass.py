import numpy as np
from matplotlib.pyplot import figure, show
from sklearn.ensemble import RandomForestClassifier
from ExternFunctions import dbplot, dbprobplot
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd

#Data impoting
kaggle_file_path = "C:/Users/MLLT/COWI/A273512 - Health evaluation of Power Transformers - Documents/20-Basis/80-KAGGLE dataset/Health index2.csv"
kaggle_data = np.genfromtxt(kaggle_file_path, delimiter=',')
kaggle_data = kaggle_data[1:]

kaggle_attributeNames = ["H2","O2","N","CH4","CO", "CO2","C2H4","C2H6","C2H2","DBDS","PF","Int V", "Diel Rig", "H2O"]
kaggle_label_name = ["Health Index"]


#Data Handling
kaggle_X = kaggle_data[:,:9]
kaggle_y = kaggle_data[:,14]

#classifying labels
boundaries = [0, 30, 50, 70, 85]
bin_indices = np.digitize(kaggle_y, boundaries)
y_class = np.where(bin_indices == 0, 0, bin_indices)
y_class = np.where(bin_indices == len(boundaries), len(boundaries), y_class)
y_class = np.ravel(y_class)

# Select one random datapoint from each class
unique_classes = np.unique(y_class)  # Unique classes derived from boundaries

for cls in unique_classes:
    class_indices = np.where(y_class == cls)[0]  # Indices where the class label is cls
    selected_index = np.random.choice(class_indices)  # Randomly select one index
    selected_datapoint = kaggle_X[selected_index, :]  # Extract the datapoints

    # Format the output as a string for clearer reading
    datapoint_str = ", ".join(f"{attr}: {val:.2f}" for attr, val in zip(kaggle_attributeNames[:9], selected_datapoint))
    print(f"Class {cls}: {datapoint_str}")
    
    
N, M = kaggle_X.shape
C = len(kaggle_label_name)

all_y_test = []
all_y_pred = []

K = 10
cv_outer = KFold(n_splits=K, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
inner_results = []
for train_ix, test_ix in cv_outer.split(kaggle_X):
    # split data
    X_train, X_test = kaggle_X[train_ix, :], kaggle_X[test_ix, :]
    y_train, y_test = y_class[train_ix], y_class[test_ix]
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(random_state=1, class_weight="balanced")
    # define search space
    space = dict()
    space['n_estimators'] = [10, 100, 1000]
    space['max_features'] = [2, 4, 8]
    # define search
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    # evaluate the model
    acc = accuracy_score(y_test, yhat)
    # store the result
    outer_results.append(acc)
    
    all_y_test.extend(y_test)
    all_y_pred.extend(yhat)
    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    # Collecting inner results for plots
    means = result.cv_results_['mean_test_score']
    params = result.cv_results_['params']
    for mean, param in zip(means, params):
        inner_results.append((mean, param))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(all_y_test, all_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.arange(1,6), yticklabels=np.arange(1,6))
plt.title('Confusion Matrix for Random Forest across All Folds')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting outer fold accuracies
plt.figure(figsize=(10, 6))
plt.plot(outer_results, marker='o', linestyle='-', markersize=8)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Outer Fold Accuracies in Random Forest Model')
plt.xticks(range(K), [f"Fold {i+1}" for i in range(K)])
plt.show()

# Inner results heatmap
param_performance = pd.DataFrame([{
    **param, 'score': mean} for mean, param in inner_results])

pivot_table = param_performance.pivot_table(values='score', 
                                            index=['n_estimators'], 
                                            columns=['max_features'],
                                            aggfunc='mean')
plt.figure(figsize=(9, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title('Inner Fold Parameter Tuning Performance')
plt.show()

#Fitting best model
best_model = RandomForestClassifier(**search.best_params_, random_state=1)
best_model.fit(kaggle_X, y_class)

#Confusion matrix
CM = confusion_matrix(y_train, best_model.predict(X_train))

#Plotting developement of health index over DGA period
X_Landerupgaard = np.array([[10, 4190, 12515, 1, 6, 99, 0, 0, 0],[10, 13877, 46206, 6, 139, 601, 1, 0, 1],[10, 4473, 11741 ,9, 137, 497, 1, 0, 1],[10, 4475, 12973, 8, 186, 592, 1, 0, 2],[11, 12260, 42085, 7, 222, 786, 1, 1, 2],[22, 11217, 43452, 8, 287, 856, 2, 1, 3],[10, 3418, 20231, 10, 335, 994, 2, 1, 4],[10, 4902, 22100, 10, 312, 1201, 2, 1, 4],[13, 9907, 32937,10, 381, 1120, 3,1,5],[11, 4427, 18129, 13, 474, 1546, 3, 1, 16]])
yhat_LAG = best_model.predict(X_Landerupgaard)

lag_dates = np.array(["20-06-13", "19-11-15", "27-10-16", "27-10-17", "13-08-18", "13-09-19", "12-08-20", "22-07-21", "16-08-22", "26-09-23"])

plt.plot(lag_dates, yhat_LAG)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.ylabel("Health Index Class")
plt.title("Landerupgaard class developement over time")
plt.xlabel("Date")
plt.ylim(1, 5)  # Setting the limits of y-axis from 0 to 5
plt.axhspan(4.5, 5.5, color='green', alpha=0.5,label="Very Good")
plt.axhspan(3.5, 4.5, color='teal', alpha=0.5, label="Good")
plt.axhspan(2.5, 3.5, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(1.5, 2.5, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0.5, 1.5, color='red', alpha=0.5,label="Very Poor")
plt.yticks(np.arange(1, 6, 1))  # Setting y-axis ticks to show only integers from 0 to 5
plt.legend()
plt.show()  # To display the plot

X_Ishøj = np.array([[68, 7626, 58488, 58, 383, 8837, 20, 358, 0],[10, 849, 3739, 7, 10, 132, 0, 2, 0],[33, 6072, 45713, 10, 113, 2040, 2, 11, 0],[48, 4626, 48262, 12, 210, 3127, 3, 13, 0],[10, 13027,60347, 8, 180, 3264, 3, 13, 0],[12, 2496, 70442, 13, 291, 4675, 10, 16, 0],[14, 8783, 65696, 10, 433, 7917, 58, 18, 0],[27, 3665, 61989, 13, 786, 10706, 122, 19, 0],[15, 1888, 63448, 16, 618, 10665, 170, 19, 0],[10, 11958, 55855 ,9, 481, 12734, 256, 20, 0],[19, 8371, 61520, 11, 679, 14677, 299, 21, 0],[10, 12110, 64789, 12, 570, 10704, 324, 22, 0],[15, 8632, 56246, 3, 661, 10915, 297, 14, 0],[15, 9725, 70226, 15, 873, 6998, 345, 15, 1],[17, 9879, 61573, 15, 794, 5098, 375, 18, 1],[13, 8288, 65232, 15, 673, 12159, 388, 16, 0]])
yhat_Ish = best_model.predict(X_Ishøj)

ishøj_dates = np.array(["14-12-16", "30-07-17", "24-08-17", "28-09-17", "30-10-17", "05-04-18", "29-03-19", "22-07-19", "22-04-20", "28-04-21", "21-06-21","21-04-22","30-06-22", "04-10-22", "05-12-22" ,"02-05-23"])

plt.plot(ishøj_dates, yhat_Ish)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Ishøj class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.axhspan(4.5, 5.5, color='green', alpha=0.5,label="Very Good")
plt.axhspan(3.5, 4.5, color='teal', alpha=0.5, label="Good")
plt.axhspan(2.5, 3.5, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(1.5, 2.5, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0.5, 1.5, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.ylim(1, 5)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(1, 6, 1))  # Setting y-axis ticks to show only integers from 0 to 5
plt.show()  # To display the plot

X_FGD1 = np.array([[14, 12411, 41755, 11, 445, 1595, 1, 2, 0],[17, 12158, 34928, 11, 458, 1618, 3, 2, 0],[22, 9847, 29028, 9, 374, 1735, 1, 2, 1],[23, 13835, 41946, 8, 395, 1298, 1, 1, 0],[15,9943, 36682, 8, 407, 1413, 2, 1, 0],[10, 10565, 29008, 10, 354, 1067, 1, 2, 0],[10, 10304, 25316, 11, 328, 1044, 1, 1, 0],[25, 8022, 20872, 5, 261, 1224, 0, 0, 0],[25, 17201, 36824, 4, 196, 879, 0, 15, 0],[25, 13444, 34016, 5, 211, 745, 0, 2, 0]])
yhat_FGD1 = best_model.predict(X_FGD1)

FGD1_dates = np.array(["23-08-01","16-07-03","11-11-04","25-10-16","23-08-17","20-08-18","16-09-19","24-08-20","20-07-21","02-09-22"])

plt.plot(FGD1_dates, yhat_FGD1)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde1 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.axhspan(4.5, 5.5, color='green', alpha=0.5,label="Very Good")
plt.axhspan(3.5, 4.5, color='teal', alpha=0.5, label="Good")
plt.axhspan(2.5, 3.5, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(1.5, 2.5, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0.5, 1.5, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.ylim(1, 5)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(1, 6, 1))  # Setting y-axis ticks to show only integers from 0 to 5
plt.show()  # To display the plot

X_FGD2 = np.array([[10, 14195, 49327, 10, 522, 1656, 1, 2, 0],[10, 20734, 67585, 11, 537, 1782, 1, 2, 0],[20, 16498, 53945, 8, 432, 2041, 1, 1, 0],[11, 12404, 40555, 8, 482, 1406, 1, 1, 0],[16, 15216, 50520, 7, 478, 1586, 1, 1, 0],[10, 13083, 47038, 8, 456, 1555, 2, 1, 0],[10, 11547, 37294, 9, 432, 1169, 1, 1, 0],[10, 11703, 34059, 9, 382, 1039, 3, 1, 0],[25, 11428, 31793, 4, 306, 799, 0, 0, 0],[25, 16582, 44262, 3, 231, 763, 0, 10, 0],[25, 19899, 54306, 3, 212, 641, 1, 0, 0]])
yhat_FGD2 = best_model.predict(X_FGD2)

FGD2_dates = np.array(["23-08-01","16-07-03","11-11-04","25-10-16","23-08-17","20-08-18","16-09-19","24-08-20","20-07-21","02-09-22","20-09-23"])

plt.plot(FGD2_dates, yhat_FGD2)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde2 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.axhspan(4.5, 5.5, color='green', alpha=0.5,label="Very Good")
plt.axhspan(3.5, 4.5, color='teal', alpha=0.5, label="Good")
plt.axhspan(2.5, 3.5, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(1.5, 2.5, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0.5, 1.5, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.ylim(1, 5)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(1, 6, 1))  # Setting y-axis ticks to show only integers from 0 to 5
plt.show()  # To display the plot

X_FGD3 = np.array([[11, 4427, 18129, 13, 474, 1546, 3, 1, 6],[13, 9907, 32937, 10, 381, 1120, 3, 1, 5],[10, 4902, 22100, 10, 312, 1201, 2, 1, 4],[10, 3418, 20231, 10, 335, 994, 2, 1, 4],[22, 11217, 43452, 8, 287, 856, 2, 1, 3],[11, 12260, 42085, 7, 222, 786, 1 ,1, 2],[10, 4475, 12973, 8, 186, 592, 1, 0, 2],[10, 4473, 11741, 9, 137, 497, 1, 0, 1],[10, 13877, 46206, 6, 139, 601, 1, 0, 1],[10, 4190, 12515, 1, 6, 99, 0, 0, 0]])
yhat_FGD3 = best_model.predict(X_FGD3)

FGD3_date = np.array(["20-06-13","19-11-15","27-10-16","27-10-17","13-08-18","13-09-19","12-08-20","22-07-21","16-08-22","26-09-23"])

plt.plot(FGD2_dates, yhat_FGD2)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde3 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.axhspan(4.5, 5.5, color='green', alpha=0.5,label="Very Good")
plt.axhspan(3.5, 4.5, color='teal', alpha=0.5, label="Good")
plt.axhspan(2.5, 3.5, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(1.5, 2.5, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0.5, 1.5, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.ylim(1, 5)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(1, 6, 1))  # Setting y-axis ticks to show only integers from 0 to 5
plt.show()  # To display the plot


import shap
##Insights with Shap
# Create a SHAP explainer object
pred = best_model.predict(kaggle_X)

explainer = shap.TreeExplainer(best_model)
explanation = explainer(kaggle_X)
shap_values = explanation.values

#Summarized classes
shap_values_summed = np.sum(shap_values, axis=2)

# Summarize the effects of all the features after summing across classes
shap.summary_plot(shap_values_summed, kaggle_X, feature_names=kaggle_attributeNames[:9])

# Each class individually
for i in range(5):
    class_index = i
    shap.summary_plot(shap_values[:, :, class_index], kaggle_X, feature_names=kaggle_attributeNames[:9])

# Individual force plot for a single prediction (for class_index class)
shap.force_plot(explainer.expected_value[class_index], shap_values[0, :, class_index], X_test[0, :], feature_names=kaggle_attributeNames[:9])

# Summarize the effects of all the features
shap.summary_plot(shap_values, kaggle_X, feature_names=kaggle_attributeNames[:9])

# Visualize the first prediction's explanation
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], kaggle_X[0,:], feature_names=kaggle_attributeNames[:9])

# Plot SHAP values for each feature for each sample
shap.force_plot(explainer.expected_value[1], shap_values[1], kaggle_X, feature_names=kaggle_attributeNames[:9])

# Bar plot for mean SHAP values (average impact on model output magnitude)

for i in range(5):
    class_index = i
    shap.summary_plot(shap_values[:, :, class_index], kaggle_X, plot_type="bar", feature_names=kaggle_attributeNames[:9])




