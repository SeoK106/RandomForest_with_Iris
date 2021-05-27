from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

def sample_inputs_generator(num):
    sample = [[] for _ in range(random.randint(1,5))]
    for lst in sample:
        for _ in range(num):
            lst.append(round(random.uniform(0,9),1))

    return sample

if __name__=="__main__":

    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    # 70% for training data, 30% for testing data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    # Generate and Train the RF model
    rfc = RandomForestClassifier(criterion = 'entropy', n_estimators = 8, n_jobs = -1)
    model = rfc.fit(x_train,y_train)

    # Set the threshold and Extract features(>=threshold)
    feature_threshold = SelectFromModel(model,threshold=0.1)

    #Extract selected feature values
    selected_x = feature_threshold.fit_transform(x,y)
    num = len(selected_x[0])

    #ReTrain the model with selecte data
    changed_model = model.fit(selected_x,y)
    
    feature_idx = feature_threshold.get_support()
    df = pd.DataFrame(x_test,columns=iris.feature_names)
    feature_names = df.columns[feature_idx].tolist()
    x_test = df.loc[:,feature_names]

    # Calculate the accuracy score with test data
    y_pred = changed_model.predict(x_test)
    print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

    # predict with the sample data
    sample_inputs = sample_inputs_generator(num)
    sample_outputs = model.predict(sample_inputs).tolist()
    outs = []
    for o in sample_outputs:
        outs.append(iris.target_names[o])
    result = pd.DataFrame(sample_inputs,columns=feature_names)
    result['species']=outs
    print("-"*100)
    print(result)
