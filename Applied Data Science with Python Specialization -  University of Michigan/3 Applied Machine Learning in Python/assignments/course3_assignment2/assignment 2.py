import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

def intro():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);

intro()


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    degree_predictions = np.zeros((4,100))
    
    degree = [1, 3, 6, 9]
    test = np.linspace(0, 10, 100)
    
    for i in range(4):
        # degree parameter
        poly = PolynomialFeatures(degree = degree[i])
        # transforms the data according to the fitted model
        X_poly = poly.fit_transform(X_train.reshape(len(X_train), 1))
        # creates an instance of the LinearRegression class with transformed training data
        linreg = LinearRegression().fit(X_poly, y_train)
        # the test array transformed according to the same polynomial features as the training data
        degree_predictions[i] = linreg.predict(poly.fit_transform(test.reshape(len(test), 1)))
    return degree_predictions


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

plot_one(answer_one())

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score

    r2_train = np.array([])
    r2_test = np.array([])
    
    # YOUR CODE HERE
    for n in range(10):
        #train polynomial linear regression
        poly = PolynomialFeatures(degree = n)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
        
        linreg = LinearRegression().fit(X_train_poly, y_train)
        
        #evaluate the polynomial linear regression
        r2_train = np.append(r2_train, linreg.score(X_train_poly, y_train))
        r2_test = np.append(r2_test, linreg.score(X_test_poly, y_test))
    return (r2_train, r2_test)

answer_two()

def plot_answer_three():
    import matplotlib.pyplot as plt
    r2_train, r2_test = answer_two()
    degrees = np.arange(0, 10)
    plt.figure()
    plt.plot(degrees, r2_train, degrees, r2_test)

plot_answer_three()

def answer_three():
    # YOUR CODE HERE
    Underfitting, Overfitting, Good_Generalization = 0, 9, 7
    return (Underfitting, Overfitting, Good_Generalization)

answer_three()

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics import r2_score
    
    poly = PolynomialFeatures(12)
    X_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))

    linreg = LinearRegression().fit(X_poly, y_train)
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)

    linlasso = Lasso(alpha=0.01, max_iter=10000, tol=0.1).fit(X_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

answer_four()


# The data in the mushrooms dataset is currently encoded with strings. 
# These values will need to be encoded to numeric to work with sklearn. 
# We'll use pd.get_dummies to convert the categorical variables into indicator variables.
mush_df = pd.read_csv('assets/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)


def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    
    feature_importances = {}
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    
    # Get the index of the top 5 inportant features
    index_feature_top5 = clf.feature_importances_.argsort()[::-1][:5]

    return X_train2.columns[index_feature_top5].tolist()

answer_five()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    
    svc = SVC(kernel='rbf', C=1, random_state=0)
    param_range = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(svc,
                                                 X_train2,
                                                 y_train2,
                                                 param_name='gamma',
                                                 param_range=param_range,
                                                 scoring='accuracy',
                                                 cv=3,
                                                 n_jobs=2)
    train_mscores, test_mscores = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

    return (train_mscores, test_mscores)

answer_six()


def plot_answer_seven():
    # computes data for the plot
    train_scores, test_scores = answer_six()    
    gamma = np.logspace(-4, 1, 6)
    
    # generates the plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(gamma, train_scores, 'b--.', label='train_scores')
    plt.plot(gamma, test_scores, 'g-*', label='test_scores')
    plt.xscale('log')  # Set x-axis scale to log 10
    plt.xticks(gamma, ['0.0001', '0.001', '0.01', '0.1', '1', '10'])  # Set custom x-axis tick labels
    plt.legend()
    

plot_answer_seven()

def answer_seven():
    param_range = np.logspace(-4, 1, 6)
    Underfitting, Overfitting, Good_Generalization = param_range[
        0], param_range[5], param_range[3]
    return (Underfitting, Overfitting, Good_Generalization)

answer_seven()