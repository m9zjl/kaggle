# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gsp
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import sklearn

data_train = pd.read_csv("/Users/ben/Documents/datasets/titanic/train.csv")


def plot_titainic(data):
    fig = plt.figure()
    fig.set(alpha=.2)

    plt.subplot(2,3,1)
    survived = data.Survived
    plt.bar(x=1,height=survived[survived==1].count())
    plt.bar(x=0,height=survived[survived==0].count())
    plt.ylabel('person count')

    plt.subplot(2,3,2)
    pclass = data.Pclass
    for i in range(1,4):
        plt.bar(x=i,height=pclass[pclass==i].count())
    plt.ylabel('plcass')

    plt.subplot(2,3,3)
    age = data.Age
    plt.scatter(data.Survived, data.Age)
    plt.grid(b=True, which='major', axis ='y')

    gs = gsp(2,3)
    plt.subplot(gs[1,:-1])
    age = data.Age
    for i in range(1,4):
        age[data.Pclass==i].plot(kind='kde')

    plt.subplot(2,3,6)
    embarked = data.Embarked
    for i in ['S','C','Q']:
        plt.bar(i,embarked[embarked==i].count())
#        plt.hist(embarked[embarked==i])

plot_titainic(data=data_train)


survived_0 = data_train.Pclass[data_train.Survived ==0].value_counts()
survived_1 = data_train.Pclass[data_train.Survived ==1].value_counts()

plt.figure()
plt.bar(range(1,survived_0.size+1),survived_0.sort_index(),label='1')
plt.bar(range(1,survived_1.size+1),survived_1.sort_index(),bottom=survived_0.sort_index(),label='2')
plt.legend()
plt.xlabel('')
plt.show()

plt.figure()
male = data_train.Survived[data_train.Sex=='male'].value_counts()
female = data_train.Survived[data_train.Sex=='female'].value_counts()
plt.bar(range(1,len(male)+1),male,label='survivied')
plt.bar(range(1,len(male)+1),female,bottom=male,label='un')
plt.legend()

plt.figure()
embarked_0 = data_train.Embarked[data_train.Survived==0].value_counts()
embarked_1=data_train.Embarked[data_train.Survived==1].value_counts()
plt.bar(['1','2','3'],embarked_0)
plt.bar(range(len(embarked_1)),embarked_1,bottom=embarked_0)
plt.xlabel('a')

g = data_train.groupby(['SibSp','Survived'])
df=pd.DataFrame(g.count()['PassengerId'])
print(df)

# feature engineering done


def set_missing_ages(df):
# 把已有的数值去除做随机森林回归，然后利用回归模型填补空缺数值
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    # 把乘客分为已知年龄和未知年龄的两组
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()
    # y is age target
    y = known_age[:,0]
    # x is feature params
    x = known_age[:,1:]
    rfr = RandomForestRegressor(random_state=42,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    # predict ages
    predictedAges = rfr.predict(unknow_age[:,1:])
    # fit missing ages with predicted ages
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df

data_train ,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


df = data_train
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1,1),age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1,1),fare_scale_param)


from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y  is target
y = train_np[:,0]
# x is feature params
x = train_np[:,1:]
# fit with logistic regression
clf = linear_model.LogisticRegression(C=1.0, tol=1e-6)
clf.fit(x,y)



data_test = pd.read_csv("/Users/ben/Documents/datasets/titanic/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1,1), fare_scale_param)


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result.to_csv('LogisticRegressionResultONTitanic.csv')









