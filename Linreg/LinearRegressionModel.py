#!/usr/bin/env python
# coding: utf-8

def module():
    import pandas as pd
    dataset=pd.read_csv('/root/50_Startups.csv')
    dataset.columns
    y=dataset['Profit']
    X=dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
    X=pd.get_dummies(X,drop_first=True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return(X_train,y_train,X_test,y_test)

def model():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import pandas as pd
    model=LinearRegression()
    X_train,y_train,X_test,y_test=module()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_test=y_test.values
    accuracy = r2_score(y_test,y_pred)
    result = accuracy*100
    print(result)
    print(type(result))
    #df1 = pd.DataFrame(result,columns=['accuracy'])
    #print(df1)
    #df1.to_csv('result.csv')
model()
