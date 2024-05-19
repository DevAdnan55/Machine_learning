# %%
import sklearn


# %%
from sklearn.datasets import load_iris

# %%
load_iris()

# %%
load_iris(return_X_y=True)

# %%
X, y = load_iris(return_X_y=True)

# %%
from sklearn.linear_model import LinearRegression

# %%
Model = LinearRegression()

# %%
Model.fit(X, y)

# %%
Model.predict(X)

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
Model_2 = KNeighborsRegressor() # Algorithm that we are using 

# %%
Model_2.fit(X, y) # Fitting the data in the algorithm 

# %%
Model_2.predict(X) # Model is predecting the y value from x 

# %%
import matplotlib.pyplot as plt
pre = Model_2.predict(X)
plt.scatter(pre,y) # Will generate a scatter plot for you 

# %%
import pandas as pd

# %%
from sklearn.datasets import fetch_openml

# %%
df = fetch_openml('titanic', version= 1, as_frame=True)['data'] # This will fetch the data for you

# %%
df.info() # Will give you the info on the data

# %%
df.isnull() # This will check if there is any null value in the data or not 

# %%
df.isnull().sum() # This will give you the totel number of null value in column 

# %%
import seaborn as sns

# %%
sns.set()
miss_val_per = pd.DataFrame((df.isnull().sum()/len(df))*100)
miss_val_per.plot(kind='bar', title='Missing value in percentage', ylabel= 'Percentage')

# %%
print(f"Size of the dataset: {df.shape}") # Diffrent type to embedded expresion in string
print("Size of the dataset:%s" %(df.shape,))
print("Size of the dataset: {}".format(df.shape))
print("Size of the dataset: " + str(df.shape))


# %%
df.drop(['body'], axis=1, inplace=True ) # If axix = 1 : Column will be removed & axix = 0 : Row will be removed, inplace = True data in df will be changed and inplace = False data wont be changed
print(f"Size of the dataset afte removing a feature: {df.shape}")


# %% [markdown]
# Data Cleaning Process

# %%
from sklearn.impute import SimpleImputer

# %%
print(f"Number of null values before imputing: {df.age.isnull().sum()} ")


# %%
imp_age = SimpleImputer(strategy= "mean") # Adds all the non null values and divide them withe the count of non null value 
df["age"] = imp_age.fit_transform(df[["age"]]) # We can create a variable for this or can use the method below to make it more readable
print(f"Number of null value in age column after imputing: {df.age.isnull().sum()} ")


# %%
df["age"] = SimpleImputer(strategy="mean").fit_transform(df[["age"]]) 
print(f"Number of null value in age column after imputing: {df.age.isnull().sum()} ")

# %% [markdown]
# Lets create a function, that can shows us all the elements in the dataset

# %%
def get_parameters(df):
    parameters = {}
    for col in df.columns[df.isnull().any()]:

        # if df[col].dtype == "float64" or df[col].dtype == "int64" or df[col].dtype == "int32" : 
        if df[col].dtypes in ["float64", 'int64', "int32"]: # Here we are checking if the columns have these types of elements or not
            strategy = "mean" # Mean is used for numerical type
        else :
            strategy = "most_frequent" # most_frequent is used for non-numerical type
        missing_value = df[col][df[col].isnull()].iloc[0]
        parameters[col] = {"missing_value": missing_value, "strategy":strategy}
    return parameters
get_parameters(df)

# %% [markdown]
# Now we are Imputing the dataset to remove null value from all the functions 

# %%
parameters =  get_parameters(df)

# %%
for col, para in parameters.items():
    strategy = para["strategy"]
    missing_value = para["missing_value"]
    
    impute = SimpleImputer(strategy=strategy, missing_values=missing_value)
    df[col] = impute.fit_transform(df[[col]]).ravel()

# %%
print(df.isnull().sum())

# %%
print(df.head())


# %%
