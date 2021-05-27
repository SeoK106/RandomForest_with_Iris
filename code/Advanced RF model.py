from sklearn import datasets,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

# 70% for training data, 30% for testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# Generate and Train the RF model
rfc = RandomForestClassifier(criterion = 'entropy', n_estimators = 8, n_jobs = -1)
model = rfc.fit(x_train,y_train)

# Calculate the accuracy score with test data
y_pred = model.predict(x_test)
print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

# predict with the sample data
sample_inputs = [[3,5,4,2],[2,3,4,5]]
sample_outputs = model.predict(sample_inputs).tolist()
for o in sample_outputs:
    print(iris.target_names[o],end=' ')
