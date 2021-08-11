import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pickle



def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size] #create value according to test_size row and column
    train_indices =  shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] #filter row and column base on integer value


if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    train , test = data_split(data,0.2)
    X_train =  train[['fiver','age','headach','runnyNose','persistent Cough','sore throatr','diffBreath','loss of smell']].to_numpy()
    X_test = test[['fiver','age','headach','runnyNose','persistent Cough','sore throatr','diffBreath','loss of smell']].to_numpy()
    Y_train = train[['innfectionProb']].to_numpy().reshape(868)
    Y_test = test[['innfectionProb']].to_numpy().reshape(217)
    clf = LogisticRegression()
    dfmodel = clf.fit(X_train,Y_train)
    input_user = [101.633515,11,0,0,1,0,0,0]
    clf.predict([input_user])
    clf.predict_proba([input_user])[0][1]
    clf.score(X_test,Y_test)

    file = open('corona_virus_sol.pkl','wb')
    pickle.dump(dfmodel,file)