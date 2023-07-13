import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7,7))
DataSet = pd.read_csv(r"C:\Users\jayson3xxx\Desktop\Training_DataSeTs\Bank Customer Churn Prediction.csv")
DataSet['gender'] = pd.factorize(DataSet['gender'])[0]
print(DataSet.info())
# print(DataSet['customer_id'].unique() , DataSet['credit_score'].unique(),
#       DataSet['country'].unique(),DataSet['gender'].unique(),
#       DataSet['tenure'].unique(),DataSet['tenure'].unique(),DataSet['churn'].unique())
DataSet['country'] = pd.factorize(DataSet['country'])[0]
DropLabels = ['customer_id']
DataSet.drop(DropLabels , axis = 1 , inplace=True)
DataSet.drop_duplicates(inplace=True)
print(DataSet.describe())
DataSet.drop(labels=['credit_score' , 'tenure' , 'products_number' ,'credit_card' , 'estimated_salary' , 'balance']
             , axis=1 , inplace=True)
X_DS = DataSet.copy();X_DS.drop(labels = ['churn'] , axis=1 , inplace=True)
sns.heatmap(DataSet.corr() , annot = True , cmap='coolwarm' )
plt.show()
Y_DS = DataSet['churn'].copy()
#X_DS = DataSet.copy();X_DS.drop(labels = ['churn','credit_score' , 'tenure'] , axis=1 , inplace=True)
Y_np = Y_DS.to_numpy()
X_np = X_DS.to_numpy()
scaler = MinMaxScaler()
X_SC = scaler.fit_transform(X_np)
x_train, x_test , y_train , y_test = train_test_split(X_np,Y_np , test_size=0.33 , random_state=42)
Model = SVC()
Model.fit(x_train , y_train)
print("правильность предсказания модели на обучающей выборке : " , Model.score(x_train,y_train))
print("правильность предсказания модели на тестовой выборке : " , Model.score(x_test,y_test))