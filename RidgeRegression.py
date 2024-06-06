import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

#Data impoting
kaggle_file_path = "C:/Users/MLLT/COWI/A273512 - Health evaluation of Power Transformers - Documents/20-Basis/80-KAGGLE dataset/Health index2.csv"
kaggle_data = np.genfromtxt(kaggle_file_path, delimiter=',')
kaggle_data = kaggle_data[1:]

kaggle_attributeNames = ["H2","O2","N","CH4","CO","C2H4","C2H6","C2H2","DBDS","PF","Int V", "Diel Rig", "H2O"]
kaggle_label_name = ["Health Index"]


#Data Handling
kaggle_X = kaggle_data[:,:9]
kaggle_y = kaggle_data[:,14:]

N, M = kaggle_X.shape
C = len(kaggle_label_name)


model = Ridge()
alpha_values = np.logspace(-6, 12, 13)  # Alpha values from 10^-6 to 10^6
param_grid = {'alpha': alpha_values}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='neg_mean_squared_error')

from sklearn.model_selection import cross_val_score

scores = cross_val_score(grid_search, kaggle_X, kaggle_y, cv=outer_cv, scoring='neg_mean_squared_error')
mean_score = np.mean(scores)

import matplotlib.pyplot as plt

grid_search.fit(kaggle_X, kaggle_y)
mean_scores = grid_search.cv_results_['mean_test_score']
std_scores = grid_search.cv_results_['std_test_score']
alphas = grid_search.cv_results_['param_alpha'].data

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, mean_scores, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Negative Mean Squared Error')
plt.title('Performance of the Ridge Regression Model')
plt.grid(True)
plt.show()

grid_search.fit(kaggle_X, kaggle_y)  # Full training
best_alpha = grid_search.best_params_['alpha']
print("Best alpha:", best_alpha)

#Fitting Final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(kaggle_X, kaggle_y)

# Coefficient Path Plot
ridge_coefficients = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(kaggle_X, kaggle_y)
    ridge_coefficients.append(model.coef_.flatten())

plt.figure(figsize=(10, 6))
for i in range(kaggle_X.shape[1]):
    plt.plot(alphas, [coef[i] for coef in ridge_coefficients], label=kaggle_attributeNames[i])

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Coefficient Path')
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot for the final model
predictions = final_model.predict(kaggle_X)
residuals = kaggle_y.flatten() - predictions.flatten()

plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()

#Plotting developement of health index
X_Landerupgaard = np.array([[10, 4190, 12515, 1, 6, 99, 0, 0, 0],[10, 13877, 46206, 6, 139, 601, 1, 0, 1],[10, 4473, 11741 ,9, 137, 497, 1, 0, 1],[10, 4475, 12973, 8, 186, 592, 1, 0, 2],[11, 12260, 42085, 7, 222, 786, 1, 1, 2],[22, 11217, 43452, 8, 287, 856, 2, 1, 3],[10, 3418, 20231, 10, 335, 994, 2, 1, 4],[10, 4902, 22100, 10, 312, 1201, 2, 1, 4],[13, 9907, 32937,10, 381, 1120, 3,1,5],[11, 4427, 18129, 13, 474, 1546, 3, 1, 16]])
yhat_LAG = final_model.predict(X_Landerupgaard)

lag_dates = np.array(["20-06-13", "19-11-15", "27-10-16", "27-10-17", "13-08-18", "13-09-19", "12-08-20", "22-07-21", "16-08-22", "26-09-23"])

plt.plot(lag_dates, yhat_LAG)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.ylabel("Health Index")
plt.title("Landerupgaard health index developement over time")
plt.xlabel("Date")
plt.ylim(0, 100)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(0, 100, 20))
plt.axhspan(85, 100, color='green', alpha=0.5,label="Very Good")
plt.axhspan(70, 85, color='teal', alpha=0.5, label="Good")
plt.axhspan(50, 70, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(30, 50, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0, 30, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.show()  # To display the plot

X_Ishøj = np.array([[68, 7626, 58488, 58, 383, 8837, 20, 358, 0],[10, 849, 3739, 7, 10, 132, 0, 2, 0],[33, 6072, 45713, 10, 113, 2040, 2, 11, 0],[48, 4626, 48262, 12, 210, 3127, 3, 13, 0],[10, 13027,60347, 8, 180, 3264, 3, 13, 0],[12, 2496, 70442, 13, 291, 4675, 10, 16, 0],[14, 8783, 65696, 10, 433, 7917, 58, 18, 0],[27, 3665, 61989, 13, 786, 10706, 122, 19, 0],[15, 1888, 63448, 16, 618, 10665, 170, 19, 0],[10, 11958, 55855 ,9, 481, 12734, 256, 20, 0],[19, 8371, 61520, 11, 679, 14677, 299, 21, 0],[10, 12110, 64789, 12, 570, 10704, 324, 22, 0],[15, 8632, 56246, 3, 661, 10915, 297, 14, 0],[15, 9725, 70226, 15, 873, 6998, 345, 15, 1],[17, 9879, 61573, 15, 794, 5098, 375, 18, 1],[13, 8288, 65232, 15, 673, 12159, 388, 16, 0]])
yhat_Ish = final_model.predict(X_Ishøj)

ishøj_dates = np.array(["14-12-16", "30-07-17", "24-08-17", "28-09-17", "30-10-17", "05-04-18", "29-03-19", "22-07-19", "22-04-20", "28-04-21", "21-06-21","21-04-22","30-06-22", "04-10-22", "05-12-22" ,"02-05-23"])

plt.plot(ishøj_dates, yhat_Ish)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.ylabel("Health Index")
plt.title("Ishøj health index developement over time")
plt.xlabel("Date")
plt.ylim(0, 100)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(0, 100, 20))  # Setting y-axis ticks to show only integers from 0 to 5
plt.axhspan(85, 100, color='green', alpha=0.5,label="Very Good")
plt.axhspan(70, 85, color='teal', alpha=0.5, label="Good")
plt.axhspan(50, 70, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(30, 50, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0, 30, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.show()  # To display the plot

X_FGD1 = np.array([[14, 12411, 41755, 11, 445, 1595, 1, 2, 0],[17, 12158, 34928, 11, 458, 1618, 3, 2, 0],[22, 9847, 29028, 9, 374, 1735, 1, 2, 1],[23, 13835, 41946, 8, 395, 1298, 1, 1, 0],[15,9943, 36682, 8, 407, 1413, 2, 1, 0],[10, 10565, 29008, 10, 354, 1067, 1, 2, 0],[10, 10304, 25316, 11, 328, 1044, 1, 1, 0],[25, 8022, 20872, 5, 261, 1224, 0, 0, 0],[25, 17201, 36824, 4, 196, 879, 0, 15, 0],[25, 13444, 34016, 5, 211, 745, 0, 2, 0]])
yhat_FGD1 = final_model.predict(X_FGD1)

FGD1_dates = np.array(["23-08-01","16-07-03","11-11-04","25-10-16","23-08-17","20-08-18","16-09-19","24-08-20","20-07-21","02-09-22"])

plt.plot(FGD1_dates, yhat_FGD1)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde1 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.ylim(0, 100)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(0, 100, 20))  # Setting y-axis ticks to show only integers from 0 to 5
plt.axhspan(85, 100, color='green', alpha=0.5,label="Very Good")
plt.axhspan(70, 85, color='teal', alpha=0.5, label="Good")
plt.axhspan(50, 70, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(30, 50, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0, 30, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.show()  # To display the plot

X_FGD2 = np.array([[10, 14195, 49327, 10, 522, 1656, 1, 2, 0],[10, 20734, 67585, 11, 537, 1782, 1, 2, 0],[20, 16498, 53945, 8, 432, 2041, 1, 1, 0],[11, 12404, 40555, 8, 482, 1406, 1, 1, 0],[16, 15216, 50520, 7, 478, 1586, 1, 1, 0],[10, 13083, 47038, 8, 456, 1555, 2, 1, 0],[10, 11547, 37294, 9, 432, 1169, 1, 1, 0],[10, 11703, 34059, 9, 382, 1039, 3, 1, 0],[25, 11428, 31793, 4, 306, 799, 0, 0, 0],[25, 16582, 44262, 3, 231, 763, 0, 10, 0],[25, 19899, 54306, 3, 212, 641, 1, 0, 0]])
yhat_FGD2 = final_model.predict(X_FGD2)

FGD2_dates = np.array(["23-08-01","16-07-03","11-11-04","25-10-16","23-08-17","20-08-18","16-09-19","24-08-20","20-07-21","02-09-22","20-09-23"])

plt.plot(FGD2_dates, yhat_FGD2)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde2 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.ylim(0, 100)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(0, 100, 20))  # Setting y-axis ticks to show only integers from 0 to 5
plt.axhspan(85, 100, color='green', alpha=0.5,label="Very Good")
plt.axhspan(70, 85, color='teal', alpha=0.5, label="Good")
plt.axhspan(50, 70, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(30, 50, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0, 30, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.show()  # To display the plot

X_FGD3 = np.array([[11, 4427, 18129, 13, 474, 1546, 3, 1, 6],[13, 9907, 32937, 10, 381, 1120, 3, 1, 5],[10, 4902, 22100, 10, 312, 1201, 2, 1, 4],[10, 3418, 20231, 10, 335, 994, 2, 1, 4],[22, 11217, 43452, 8, 287, 856, 2, 1, 3],[11, 12260, 42085, 7, 222, 786, 1 ,1, 2],[10, 4475, 12973, 8, 186, 592, 1, 0, 2],[10, 4473, 11741, 9, 137, 497, 1, 0, 1],[10, 13877, 46206, 6, 139, 601, 1, 0, 1],[10, 4190, 12515, 1, 6, 99, 0, 0, 0]])
yhat_FGD3 = final_model.predict(X_FGD3)

FGD3_date = np.array(["20-06-13","19-11-15","27-10-16","27-10-17","13-08-18","13-09-19","12-08-20","22-07-21","16-08-22","26-09-23"])

plt.plot(FGD3_date, yhat_FGD3)  # Plotting the predicted values
plt.xticks(rotation=90)  # Rotating the x-axis labels for better visibility
plt.title("Fraugde3 class developement over time")
plt.ylabel("Health Index Class")
plt.xlabel("Date")
plt.ylim(0, 100)  # Setting the limits of y-axis from 0 to 5
plt.yticks(np.arange(0, 100, 20))  # Setting y-axis ticks to show only integers from 0 to 5
plt.axhspan(85, 100, color='green', alpha=0.5,label="Very Good")
plt.axhspan(70, 85, color='teal', alpha=0.5, label="Good")
plt.axhspan(50, 70, color='yellow', alpha=0.5, label="Fair")
plt.axhspan(30, 50, color='orange', alpha=0.5,label="Poor")
plt.axhspan(0, 30, color='red', alpha=0.5,label="Very Poor")
plt.legend()
plt.show()  # To display the plot



#Insights with Shap
import shap

explainer = shap.LinearExplainer(final_model, kaggle_X)
shap_values = explainer.shap_values(kaggle_X)

shap.summary_plot(shap_values, kaggle_X, feature_names=kaggle_attributeNames[:9])
shap.summary_plot(shap_values, kaggle_X, plot_type="bar", feature_names=kaggle_attributeNames[:9])


