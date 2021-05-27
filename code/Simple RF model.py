from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Generate and Train the RF model
rfc = RandomForestClassifier(criterion = 'entropy', #default='gini'
                             n_estimators = 8, #number of trees
                             n_jobs = -1) #number of processors to run in parallel('-1' means 'using all')
model = rfc.fit(x,y)

# Predict with the learned model
sample_inputs = [[5,4,3,2],[1,3,2,5]]
predictions = model.predict(sample_inputs) #return array([[1],[2]])
predictions = predictions.tolist()
for p in predictions:
    print(iris.target_names[p],end=' ')
