# Importing necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#preprocessing before to avoid repeating code 
def preprocess(df):
    areacols_to_sum = ['TotalBsmtSF','1stFlrSF','2ndFlrSF', 'GarageArea', 'PoolArea', 'WoodDeckSF', 'ScreenPorch', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch']
    squarefootage = df[areacols_to_sum].sum(axis=1)

    nbath = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    nbathroom = df[nbath].sum(axis=1)
    
    IndepVar = pd.DataFrame({ 'Bedroom' : df['BedroomAbvGr'],
        'Bathroom' : nbathroom,
        'Area' : squarefootage})
    
    return IndepVar


# reading the train.csv file to get required training data
# used train_test_split to test the trained model too
def training():
    data = pd.read_csv("https://github.com/Vevelugu/PRODIGY_ML_01/blob/main/house-prices-advanced-regression-techniques/train.csv?raw=true")
    df = pd.DataFrame(data)
    
    IndepVar = preprocess(df)
    DepVar = df['SalePrice']
    
    IndepVartr, IndepVarts, DepVartr, DepVarts = train_test_split(IndepVar, DepVar, test_size=0.2)

    lr = LinearRegression()
    lr.fit(IndepVartr, DepVartr)
     
    R2_score = lr.score(IndepVarts, DepVarts)
    print(f"R2 Score of the model is {R2_score*100}%")
    
    return lr

# the testing function using data from test.csv
def predictions(lr):
    data = pd.read_csv("https://github.com/Vevelugu/PRODIGY_ML_01/blob/main/house-prices-advanced-regression-techniques/test.csv?raw=true")
    df = pd.DataFrame(data)
    
    IndepVar = preprocess(df)
    DepVar = pd.DataFrame(index=df['Id'])
    DepVar['SalesPrice'] = lr.predict(IndepVar).round(2)
    
    return DepVar

lr = training()
print(f"Predicted Values of Sales are \n{predictions(lr)}")
