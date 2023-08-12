import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv("Final.csv", encoding = 'unicode_escape', index_col=0)
data.drop(['Player'], axis=1, inplace=True)
data.drop(['Club_x'], axis=1, inplace=True)
data.drop(['Min'], axis=1, inplace=True)
data.drop(['PK_x'], axis=1, inplace=True)
data.drop(['PKatt_x'], axis=1, inplace=True)
data.drop(['CrdY'], axis=1, inplace=True)
data.drop(['CrdR'], axis=1, inplace=True)
data.drop(['Gls90'], axis=1, inplace=True)
data.drop(['Ast90'], axis=1, inplace=True)
data.drop(['G+A'], axis=1, inplace=True)
data.drop(['Gls'], axis=1, inplace=True)
data.drop(['Ast'], axis=1, inplace=True)
data.drop(['PKatt'], axis=1, inplace=True)
data.drop(['Sh'], axis=1, inplace=True)
data.drop(['SoT'], axis=1, inplace=True)
data.drop(['SoT%'], axis=1, inplace=True)
data.drop(['Sh/90'], axis=1, inplace=True)
data.drop(['SoT/90'], axis=1, inplace=True)
data.drop(['G/Sh'], axis=1, inplace=True)
data.drop(['G/SoT'], axis=1, inplace=True)
data.drop(['TakleD'], axis=1, inplace=True)
data.drop(['Tkl%'], axis=1, inplace=True)
data.drop(['Press'], axis=1, inplace=True)
data.drop(['Succ_x'], axis=1, inplace=True)
data.drop(['%'], axis=1, inplace=True)
data.drop(['Blocks'], axis=1, inplace=True)
data.drop(['ShotB'], axis=1, inplace=True)
data.drop(['PassB'], axis=1, inplace=True)
data.drop(['Int'], axis=1, inplace=True)
data.drop(['Clr'], axis=1, inplace=True)
data.drop(['Passes Completed'], axis=1, inplace=True)
data.drop(['Cmp%'], axis=1, inplace=True)
data.drop(['Touches'], axis=1, inplace=True)
data.drop(['Succ_y'], axis=1, inplace=True)
data.drop(['Att'], axis=1, inplace=True)
data.drop(['Succ%'], axis=1, inplace=True)
data.drop(['#Pl'], axis=1, inplace=True)

data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data.interpolate(method ='linear', limit_direction ='forward', inplace=True)
data["Leauge"].fillna(data["Leauge"].mode()[0], inplace=True)

Leauge = {'Premier Leauge': 1, 'La Liga': 2, "Bundesliga" :3, "Serie A": 4, "Ligue1": 5 }
Pos = {'MF': 1, 'DF': 2, "FW" :3, "WB": 4 }
data.Leauge = [Leauge[item] for item in data.Leauge]
data.Pos = [Pos[item] for item in data.Pos]

print(list(data.columns))

y = data.iloc[:,0:1].values
x = data.iloc[:,1:].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

reg = LinearRegression()
lr = reg.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#print(X_test)
print("Mean squared error :",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 score :",r2_score(y_test, y_pred))

# pickle.dump(lr, open("Market_Value.pkl", "wb"))
# model = pickle.load((open("Market_Value.pkl", "rb")))

arr = np.array([[1,2,19,125,100,29,9,6,55,40,3000]])

y_new = lr.predict(arr)
print("The predicted Market Value is :",y_new[0][0])
