import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

# Load from datasets
df = pd.read_csv('Fish.csv')

nRows, nCols = df.shape
print(f'{nRows} rows and {nCols} columns')

df.describe()

# Correlation
def correlation(data):
    df_correlation = data.corr()
    ax = sns.heatmap(
        df_correlation, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    return df_correlation

#correlation(df)

x = df[['Species','Length1','Length2','Length3','Height','Width']]
y = df.Weight

label_encoder = LabelEncoder()
x.iloc[:,0] = label_encoder.fit_transform(x.iloc[:,0])
hot_encoder = OneHotEncoder()
x = hot_encoder.fit_transform(x).toarray()


def multiple_linear_regression():       
    
    # Splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 1)
        
    # Fitting
    reg_function = LinearRegression()
    reg_function.fit(x_train, y_train)
    print("\n Linear regression coefficients: ", reg_function.intercept_, reg_function.coef_, '\n')
        
    # Prediction
    prediction = reg_function.predict(x_test)
    print("Prediction for model: ", prediction, '\n')
        
    # Metrics
    print("Absolute error: ", metrics.mean_absolute_error(y_test, prediction), '\n')
    print("Mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test,prediction)), '\n')
    print("R square: ", r2_score(y_test,prediction), '\n')   

    plt.scatter(y_test, prediction, color='black')
    plt.xlabel('Fishes weight')
    plt.ylabel('Predicted weight')
    plt.plot(y_test, y_test + 1, linestyle='solid', color='red')
    plt.grid()
    plt.show()
    
multiple_linear_regression()
