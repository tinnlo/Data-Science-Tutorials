# Q5

# get decision functions scores 
y_scores_m = m.fit(X_train, y_train).decision_function(X_test)
# get precision, recall and thresholds values
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_m)

# Find the precision corresponding to a recall of 0.8
target_recall = 0.8
precision_at_target_recall = np.interp(target_recall, recall[::-1], precision[::-1])

#print('Estimated Precision at Recall of {:.2f}: {:.2f}'.format(recall_target, precision_at_target_recall))

# show the precision-recall curve in the plot
plt.plot(precision, recall)
plt.scatter(target_recall, precision_at_target_recall, color='red', marker='o', 
            label='Recall={:.2f}, Precision={:.2f}'.format(target_recall, precision_at_target_recall))

plt.xlabel('precision')
plt.ylabel('recall')
plt.legend(loc='lower left')
plt.show()


# Q8

m.fit(X_train,y_train)
y_predicted = m.predict(X_test)
print(precision_score(y_test,y_predicted, average = 'macro'))

# Q13 & 14

# Grid search for recall
grid_values = {'gamma':[0.01,0.1,1,10],'C':[0.01,0.1,1,10]}
grid_clf_acc = GridSearchCV(m, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)
print(grid_clf_acc.best_params_)
print(grid_clf_acc.best_score_)

clf_new = SVC(kernel='rbf', C=0.01, gamma=0.01).fit(X_train, y_train)

y_predicted = clf_new.predict(X_test)

print(recall_score(y_test, y_predicted) - precision_score(y_test, y_predicted))

# Grid search for precision
grid_values = {'gamma':[0.01,0.1,1,10],'C':[0.01,0.1,1,10]}
grid_clf_acc = GridSearchCV(m, param_grid = grid_values,scoring = 'precision')
grid_clf_acc.fit(X_train, y_train)
print(grid_clf_acc.best_params_)

clf_new = SVC(kernel='rbf', C=10, gamma=1).fit(X_train, y_train)

y_predicted = clf_new.predict(X_test)

print(precision_score(y_test, y_predicted) - recall_score(y_test, y_predicted) )
