from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def select_features(features,num):
    top = features[:num].index.tolist()
    
    tmp = []
    for a in top:
        tmp.append( ' '.join(a.split(' ')[:2]))
    top = tmp

    return top

def visualize_feature_importance(features):
    # Generate a bar plot
    sns.barplot(x=features, y=features.index)

    # Add labels to the graph of features
    plt.xlabel('Importance Score of Features')
    plt.ylabel('Features')
    plt.title("Visualizing Features and Imp_scores")
    plt.show()

def sample_inputs_generator(num):
    sample = [[] for _ in range(random.randint(1,5))]
    for lst in sample:
        for _ in range(num):
            lst.append(round(random.uniform(0,9),1))

    return sample

if __name__=="__main__":
    iris = datasets.load_iris()
    
    data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
    })

    x = iris.data
    y = iris.target

    # 70% for training data, 30% for testing data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    # Generate and Train the RF model
    rfc = RandomForestClassifier(criterion = 'entropy', n_estimators = 8, n_jobs = -1)
    model = rfc.fit(x_train,y_train)

    #Calculate feature importance in descending order
    ft_importances = pd.Series(model.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
    visualize_feature_importance(ft_importances)

    #Generate the datasets Again
    num = 3
    top = select_features(ft_importances,num)
    x = data[top]

    # Resetting training and testing data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    model = rfc.fit(x_train,y_train)

    # Calculate the accuracy score with test data
    y_pred = model.predict(x_test)
    print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

    # predict with the sample data
    sample_inputs = sample_inputs_generator(num)
    sample_outputs = model.predict(sample_inputs).tolist()
    outs = []
    for o in sample_outputs:
        outs.append(iris.target_names[o])
    result = pd.DataFrame(sample_inputs,columns=top)
    result['species']=outs
    print("-"*100)
    print(result)
