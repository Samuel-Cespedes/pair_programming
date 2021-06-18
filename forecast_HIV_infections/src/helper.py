import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

def fit_predict_plot(df, features, response):
    X = df[features]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    model = LinearRegression()
    model.fit(X_train, y_train)

    yhat = model.predict(X_test)

    plt.scatter(yhat, y_test - yhat);
    return f'RMSE = {np.sqrt(mean_squared_error(y_test, yhat))}'