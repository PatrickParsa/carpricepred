#!/usr/bin/env python
# coding: utf-8

# # Honda Civic price prediction modeling
# 
# ---
# 
# ## Patrick Parsa
# 
# ### Dataset scraped from local car website

# **Importing packages and initial data overview**

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_excel('/Users/patrick/Documents/ML model deployment/multi_page_car.xlsx')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


#missing values
df.isna().sum()


# ## Data cleaning and feature engineering
# ---

# ### Steps:
# 
# * Filtering to only contain desired models (Civic, Accord, Corolla, Camry)
# * Filtering out fully-optional models
# * Converting Price column into integer and removing dollar sign and comma
# * Converting mileage into integer and removing 'mi' and comma
# * Creating new column 'Year'
# * Converting 'Year' into an integer and then using that to create new column 'Age'

# Creating new column: **Year**

# In[8]:


df['Year'] = df.Name.str[:4]


# In[9]:


df.head()


# Creating new column: **Age**

# In[10]:


df['Year'] = df['Year'].astype(str).astype(int)


# In[11]:


df.dtypes


# In[12]:


df['Age'] = 2022 - df['Year']
df


# **Removing unecessary Symbols from Price and converting to int**

# In[13]:


df['Price'] = df['Price'].str.replace(',','')
df['Price'] = df['Price'].str.replace('$','')
df['Price'] = df['Price'].astype(int)


# In[14]:


df.head()


# **Same process for mileage** 

# In[15]:


df['Mileage'] = df['Mileage'].str.replace(',','')
df['Mileage'] = df['Mileage'].str.replace(' mi.','')
df.head()


# In[16]:


df['Mileage'] = df['Mileage'].astype(int)
df.dtypes


# Saving df before filtering to only include targeted models, in case we want to build a project with other makes and models in the future. 

# In[17]:


original_df = df


# **Categorizing for targeted models (Civic, Accord, Corolla, Camry)**

# In[18]:


df.loc[df['Name'].str.contains('Civic'), 'Model'] = 'Civic'
df.loc[df['Name'].str.contains('Accord'), 'Model'] = 'Accord'
df.loc[df['Name'].str.contains('Corolla'), 'Model'] = 'Corolla'
df.loc[df['Name'].str.contains('Camry'), 'Model'] = 'Camry'
df.head()


# In[19]:


df = df.dropna()


# In[20]:


df.head()


# In[21]:


df.Name.unique()


# **Filtering out full-options models**

# In[22]:


#For Civic
df.loc[df['Name'].str.contains('Si'), 'Full options'] = 'Yes'

#For Accord
df.loc[df['Name'].str.contains('Touring'), 'Full options'] = 'Yes'
df.loc[df['Name'].str.contains('Touring 2.0'), 'Full options'] = 'Yes'
df.loc[df['Name'].str.contains('Sport 2.0'), 'Full options'] = 'Yes'

#For Corolla
df.loc[df['Name'].str.contains('XSE'), 'Full options'] = 'Yes'

#For Camry
df.loc[df['Name'].str.contains('XLE'), 'Full options'] = 'Yes'

#filtering out hybrids
df.loc[df['Name'].str.contains('Hybrid'), 'Full options'] = 'Yes'


# In[23]:


df.head()


# In[24]:


df = df.loc[(df['Full options'] != 'Yes')]


# In[25]:


df.head()


# In[26]:


df['Model'].value_counts()


# In[27]:


df = df.drop('Full options',1)


# In[28]:


df.head()


# **Statistical Summary**

# In[29]:


df.describe()


# ## EDA

# In[30]:


final_dataset = df[['Mileage','Price','Model','Age']]
final_dataset.head()


# In[31]:


final_dataset = final_dataset.reset_index()


# In[32]:


final_dataset.head()


# In[33]:


final_dataset = final_dataset.drop('index',1)


# In[34]:


final_dataset.head()


# ### Univariate Analysis

# **Categorical columns:**

# In[35]:


import seaborn as sns


# In[36]:


sns.set_theme(style = 'darkgrid')
ax = sns.countplot(x='Model',data=final_dataset)


# **Numerical columns**

# In[37]:


num_cols = ['Mileage','Price','Age']
i=0
while i < 3:
    fig = plt.figure(figsize=[20,3])
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    
    #ax1.title.set_text(num_cols[i])
    plt.subplot(1,2,1)
    sns.boxplot(x=num_cols[i], data=final_dataset)
    i += 1
    
    
    
    plt.show()


# **Bivariate/Multi-Variate Analysis**

# In[38]:


sns.heatmap(final_dataset.corr(), annot=True, cmap="RdBu")
plt.show()


# In[39]:


final_dataset.corr()['Price']


# In[40]:


sns.pairplot(final_dataset)


# ## Modeling

# In[41]:


backup = final_dataset


# In[42]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[43]:


final_dataset.head()


# Convert to csv

# In[65]:


final_dataset.to_csv('cars_real.csv')


# In[44]:


first_column = final_dataset.pop('Price')
final_dataset.insert(0,'Price',first_column)
final_dataset.head()


# In[45]:


##independent and dependent features
X = final_dataset.iloc[:,1:]
y= final_dataset.iloc[:,0]
X.head()


# In[46]:


y.head()


# In[47]:


### Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor() 
model.fit(X,y)


# In[48]:


print(model.feature_importances_)


# In[49]:


#Graph of feature importances
feat_importances = pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[51]:


print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# ### Model Creation/Evaluation

# **Applying regression models**
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 5. Gradient Boosting regression

# In[52]:


from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[53]:


CV = []
R2_train = []
R2_test = []

def car_pred_model(model,model_name):
    # Training model
    model.fit(X_train,y_train)
            
    # R2 score of train set
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train,y_pred_train)
    R2_train.append(round(R2_train_model,2))
    
    # R2 score of test set
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test,y_pred_test)
    R2_test.append(round(R2_test_model,2))
    
    # R2 mean of train set using Cross validation
    cross_val = cross_val_score(model ,X_train ,y_train ,cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean,2))
    
    # Printing results
    print("Train R2-score :",round(R2_train_model,2))
    print("Test R2-score :",round(R2_test_model,2))
    print("Train CV scores :",cross_val)
    print("Train CV mean :",round(cv_mean,2))
    
    # Plotting Graphs 
    # Residual Plot of train data
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('Residual Plot of Train samples')
    sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')
    
    # Y_test vs Y_train scatter plot
    ax[1].set_title('y_test vs y_pred_test')
    ax[1].scatter(x = y_test, y = y_pred_test)
    ax[1].set_xlabel('y_test')
    ax[1].set_ylabel('y_pred_test')
    
    plt.show()


# #### Standard Linear Regression or Ordinary Least Squares

# In[54]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
car_pred_model(lr,"Linear_regressor.pkl")


# #### Ridge

# In[55]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# Creating Ridge model object
rg = Ridge()
# range of alpha 
alpha = np.logspace(-3,3,num=14)

# Creating RandomizedSearchCV to find the best estimator of hyperparameter
rg_rs = RandomizedSearchCV(estimator = rg, param_distributions = dict(alpha=alpha))

car_pred_model(rg_rs,"ridge.pkl")


# #### Lasso

# In[56]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV

ls = Lasso()
alpha = np.logspace(-3,3,num=14) # range for alpha

ls_rs = RandomizedSearchCV(estimator = ls, param_distributions = dict(alpha=alpha))


# In[57]:


car_pred_model(ls_rs,"lasso.pkl")


# In[58]:


Technique = ["LinearRegression","Ridge","Lasso"]
results=pd.DataFrame({'Model': Technique,'R Squared(Train)': R2_train,'R Squared(Test)': R2_test,'CV score mean(Train)': CV})
display(results)


# In[59]:


lr.fit(X_train,y_train)


# In[60]:


predictions = lr.predict(X_test)


# In[61]:


predictions


# In[62]:


sns.histplot(y_test-predictions)


# In[63]:


plt.scatter(y_test,predictions)


# In[66]:


import pickle
#opening a file, where I want to store the data
file = open('linear_regression_model1.pkl','wb')

#dumping information to that file
pickle.dump(lr,file)


# In[68]:


pickle.dump(lr,open('lr_model.pkl','wb'))

