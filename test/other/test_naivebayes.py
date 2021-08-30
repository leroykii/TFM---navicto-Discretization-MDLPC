from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Example based on: https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

def main():
    
    #Load dataset
    wine = datasets.load_wine()
    print("Features: ", wine.feature_names)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109) # 70% training and 30% test


    # Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #model.fit(features,label)
    
if __name__ == '__main__':
    main()