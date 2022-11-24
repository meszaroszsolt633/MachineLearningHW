from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from joblib import load,dump
from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
import numpy as np
import sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import sklearn
import seaborn as sns
sns.set(style="white")
sns.set(style="darkgrid", color_codes=True)


if __name__ == '__main__':
    penguins = pd.read_csv('D:\egyetem\penguins_size.csv')
    penguins = penguins.dropna(how='any')
    penguins.head()

    #encoding
    le = LabelEncoder()
    penguins.iloc[:, 0] = le.fit_transform(penguins.iloc[:, 0])
    penguins.iloc[:, 1] = le.fit_transform(penguins.iloc[:, 1])
    penguins.iloc[:, 6] = le.fit_transform(penguins.iloc[:, 6])


    predictors = ['sex', 'species', 'island', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = penguins[predictors]
    y = penguins['culmen_length_mm']


    poly_feat = PolynomialFeatures(degree=2)
    X_poly = poly_feat.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    svr_model = SVR(kernel='linear')
    svr_model.fit(X_train,y_train)
    svr_y_pred = svr_model.predict(X_test)
    svr_mae = mean_absolute_error(y_test, svr_y_pred)
    print(svr_mae)

    svr_rmse = np.sqrt(mean_squared_error(y_test, svr_y_pred))
    print(svr_rmse)

    svr_r2 = r2_score(y_test, svr_y_pred)
    print(svr_r2)





    #SVMReg = SVR(kernel='rbf', C=10, epsilon=0.1,gamma=1e-18)
    #SVMReg.fit(X_train, y_train)

    dump(svr_model, 'SVR_Model.joblib')
    dump(X_poly, 'poly_conv.joblib')
    dump(scaler, 'std_scaler.joblib')




