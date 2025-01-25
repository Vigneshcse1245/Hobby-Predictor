#import required modules
import pandas as pd
import sklearn.linear_model as sklm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

#making dataframes
df = pd.read_csv('training_data.csv')
dftest = pd.read_csv('test_data.csv')

#creating encoder objects
LE = LabelEncoder()
OHE = OneHotEncoder(sparse_output=False)


#parallely preprocessing test and training data
dftest.drop('ID',axis=1,inplace=True)

df.replace({'yes':'1','no':'-1','maybe':'0'},inplace=True)

dftest.replace({'yes':'1','no':'-1','maybe':'0'},inplace=True)

#replaces all occurances of yes,maybe,no with 1,0,-1 respectively in the categories given below
to_replace = ["Olympiad_Participation",'Scholarship','School','Projects','Medals','Career_sprt','Act_sprt','Fant_arts','Won_arts']
for i in to_replace:
    df[i] = LE.fit_transform(df[i])

    dftest[i] = LE.fit_transform(dftest[i])

#replaces the subjects with numbers
new = OHE.fit_transform(df[['Fav_sub']])

newtest = OHE.fit_transform(dftest[['Fav_sub']])

#labels the columns replaced subjects by the name
newcols= pd.DataFrame(new,columns=OHE.get_feature_names_out())

newtestcols= pd.DataFrame(newtest,columns=OHE.get_feature_names_out())

#add the new columns to the dataframe
df = pd.concat([df,newcols],axis=1)

dftest = pd.concat([dftest,newtestcols],axis=1)

#dropping unneccesary columns, we can drop a dummy column as all other columns zero (in categories of fav subject)
#implies 1 in the last remaining column hence we can drop that column
df.drop(["Fav_sub","Fav_sub_Any language"],axis=1,inplace=True)

dftest.drop(["Fav_sub","Fav_sub_Any language"],axis=1,inplace=True)

#final dataframe to fit
X = df.drop("Predicted_Hobby",axis=1)
y = LE.fit_transform(df['Predicted_Hobby'])


# Logistic Regression
LogiReg = sklm.LogisticRegression()
LogiReg.fit(X,y)


#linear regression
le=sklm.LinearRegression()
le.fit(X,y)


# decision tree
DecTree = tree.DecisionTreeClassifier()
DecTree.fit(X,y)

#svm
SVM = svm.SVC()
SVM.fit(X,y)

#random forest
RandomFr = RandomForestClassifier(n_estimators=500)
RandomFr.fit(X,y)

#creating pipelines
LOGREG_pipeline = Pipeline([('model',LogiReg)])
DecTree_pipeline = Pipeline([('model',DecTree)])
SVM_pipeline = Pipeline([('model',SVM)])
RanTree_pipeline = Pipeline([('model',RandomFr)])

#taking the best model by comparing mean of cross validate score 
score = 0

pipelines = [LOGREG_pipeline,DecTree_pipeline,SVM_pipeline,RanTree_pipeline]
for pipeline in pipelines:
    prev_score = score
    score = cross_val_score(pipeline,X,y,cv=5,scoring='accuracy').mean()
    if score >= prev_score:
        a,b = score,pipeline

print(f"the best model from the above models is {b['model']} with a mean score of {a}")

Best_pipeline = b

#predicting hobby of test data and Re-encode to human readable form
predicted_hobby = Best_pipeline.predict(dftest)

#this compresses to 1D array
predicted_hobby = predicted_hobby.ravel()

predicted_hobby = pd.DataFrame(predicted_hobby)

predicted_hobby["predicted_hobby"] = predicted_hobby[0]
predicted_hobby.drop(0,axis=1,inplace=True)

#converting to human readable dataframe
predicted_hobby.replace({0:'Academics',1:'Arts',2:'Sports'},inplace=True)

Predicted_data = pd.concat([pd.read_csv('test_data.csv'),predicted_hobby],axis=1)

Predicted_data.to_csv('predicted_data.csv',index=False)

#save the best model
joblib.dump(Best_pipeline,'Best_trained_model.pkl')