#!/usr/bin/env python
# coding: utf-8

# In[28]:


#Importing libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# In[29]:


Airbnb = pd.read_csv('Airbnb_dataset.csv')


# In[24]:


Airbnb.head()


# In[25]:


Airbnb.tail()


# In[26]:


Airbnb.shape


# In[27]:


Airbnb.info()


# In[7]:


Airbnb.describe().T


# In[8]:


Airbnb.isnull().sum().sort_values(ascending=False)


# #### Observations
# 
# 1. Only 3 of the 11 variables do not have missing values. The review score rating column has the largest number of missing values, at 16,722
# 
# 2. We do not seem to be dealing with much skewness in this date. The mean and 50% quartile is close for almost all the variables
# 
# 3. We are seeing zero values in variables like bedroom, beds and bathroom. This is assumed to be an error and would be left as is because that was not yet covered at the stage of learning this project relates to

# In[10]:


Airbnb.nunique().sort_values(ascending=False)


# #### Dropping columns
# We will be dropping columns to make the data more useable. The columns we are dropping now are the 'id' and 'review_scores_rating'. 
# 
# The 'id' column would be dropped because it is not so relevant to our analysis
# 
# The 'review_scores_rating' column because of the large number of missing values. Imputing would greatly distort the data, so we just drop it.

# In[30]:


#dropping id column
Airbnb.drop(columns=['id'], inplace=True)


# In[31]:


#dropping review_scores_rating column
Airbnb.drop(columns=['review_scores_rating'], inplace=True)


# In[32]:


#fixing data types
cols = Airbnb.select_dtypes(['object'])
cols.columns


# In[33]:


for i in cols.columns:
    Airbnb[i] = Airbnb[i].astype('category')


# ### Let's now see what the data looks like

# In[15]:


Airbnb.head()


# In[16]:


Airbnb.info()


# In[17]:


# filtering object type columns
cat_columns = Airbnb.describe(include=['category']).columns
cat_columns


# In[18]:


for i in cat_columns:
    print('Unique values in',i, 'are :')
    print(Airbnb[i].value_counts())
    print('*'*50)


# ### Data Prepocessing

# #### Imputing missing values

# In[67]:


#let us see the type of missing values that we have
Airbnb.sample(10)


# What can be seen here is NaN but on a closer look at the raw csv data, other missing values are just blank.
# 
# There are two options here, impute any of mean, median or mode into the NaNs, then do later for the blanks. Or fill the blanks with NaN and do the imputation of mean/meadian/mode for all NaNs.
# 
# We will go with the former. 

# In[68]:


Airbnb.median()


# In[21]:


Airbnb.mean()


# In[22]:


Airbnb.mode()


# Mode and median have similar numbers for numerical values, and since the mode has values for the columns with categorical values, we will go with filling missing values with the median of each column. Then fill the categorical columns with the modal values

# In[69]:


#replacing 'NaN' with each column median values
Airbnb.fillna(Airbnb.median(), inplace=True)
Airbnb.sample(n=10)


# In[70]:


Airbnb.isnull().sum().sort_values(ascending=False)


# Now we're left with column with categorical values to deal with. We will replace all the NaNs there with the mode

# #### Replacing NaNs in 'cancellation_policy', 'room_type', 'cleaning_fee' with their respective mode

# In[25]:


Airbnb['cancellation_policy'] = Airbnb['cancellation_policy'].replace('', 'strict')
Airbnb['cancellation_policy'].nunique


# In[26]:


Airbnb['room_type'] = Airbnb['room_type'].replace('', 'Entire home/apt')
Airbnb['room_type'].nunique


# In[27]:


Airbnb['cleaning_fee'] = Airbnb['cleaning_fee'].replace('', 'True')
Airbnb['cleaning_fee'].nunique


# In[28]:


#Do we still have missing values anywhere?
Airbnb.notnull().sum().sort_values(ascending=False)


# In[29]:


Airbnb.isnull().sum().sort_values(ascending=False)


# Since imputing neither median or mode did not work for these 3 features, we will be dropping the remaining lines with missing values.

# In[71]:


Airbnb1 = Airbnb.dropna(axis=0, how='any')
Airbnb1.sample(n=5)


# In[72]:


Airbnb1.isnull().sum().sort_values(ascending=False)


# We now have no missing values in any of the rows. Now moving on to EDA

# ### Exploratory Data Analysis (EDA)

# In[32]:


# We will be doing uni-variate analysis in order to study their central tendency and dispersion.
# We will write a function that will help us create boxplot and histogram for our numerical 
# variables.
# This function takes the numerical column as the input and returns the boxplots 
# and histograms for the variable.
def histogram_boxplot(feature, figsize=(15,10), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='green') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,palette="flare") if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='red', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# In[33]:


# Function to create barplots that indicate percentage for each category.

def perc_on_bar(z):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(Airbnb1[z]) # length of the column
    plt.figure(figsize=(15,5))
    #plt.xticks(rotation=45)
    ax = sns.countplot(Airbnb1[z],palette='Paired')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # height of the plot
        
        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage 
    plt.show() # show the plot


# ### Univariate Analysis

# In[34]:


perc_on_bar('room_type')


# In[35]:


histogram_boxplot(Airbnb['accommodates']);


# In[36]:


histogram_boxplot(Airbnb1['bathrooms']);


# In[37]:


perc_on_bar('cancellation_policy')


# In[38]:


perc_on_bar('cleaning_fee')


# In[39]:


perc_on_bar('instant_bookable')


# In[40]:


histogram_boxplot(Airbnb1['bedrooms']);


# In[41]:


histogram_boxplot(Airbnb1['beds']);


# In[42]:


histogram_boxplot(Airbnb1['log_price']);


# ### Multivariate Analysis
# 
# #beds vs Logprice vs bedrooms
# plt.figure(figsize=(15,7))
# sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["bedrooms"])
# plt.legend(bbox_to_anchor=(1.00, 1))
# plt.show()

# In[44]:


#beds vs Logprice vs roomtype
plt.figure(figsize=(15,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["room_type"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[45]:


#beds vs Logprice vs instant bookable
plt.figure(figsize=(15,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["instant_bookable"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[46]:


#beds vs Logprice vs cancellation policy
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["cancellation_policy"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[47]:


#beds vs Logprice vs cleaning fee
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["cleaning_fee"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[48]:


#beds vs Logprice vs bathrooms
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["bathrooms"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[80]:


#beds vs bathrooms vs instant bookable
plt.figure(figsize=(10,7))
sns.boxplot(Airbnb1["beds"],Airbnb1["bathrooms"],hue=Airbnb1["instant_bookable"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[94]:


#beds vs bathrooms vs instant bookable
plt.figure(figsize=(10,7))
sns.boxplot(Airbnb1["bedrooms"],Airbnb1["beds"],hue=Airbnb1["instant_bookable"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[93]:


#beds vs accommodates vs bathrooms 
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["accommodates"],hue=Airbnb1["bathrooms"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[88]:


#beds vs bedrooms vs bathrooms
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["bedrooms"],hue=Airbnb1["bathrooms"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[49]:


#beds vs Logprice vs accomodates
plt.figure(figsize=(10,7))
sns.stripplot(Airbnb1["beds"],Airbnb1["log_price"],hue=Airbnb1["accommodates"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[50]:


#Logprice vs instantbookablevs cancellation policy
plt.figure(figsize=(10,7))
sns.barplot(Airbnb1["instant_bookable"],Airbnb1["log_price"],hue=Airbnb1["cancellation_policy"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[51]:


#Logprice vs cancellation policy
plt.figure(figsize=(10,7))
sns.barplot(Airbnb1["cancellation_policy"],Airbnb1["log_price"],hue=Airbnb1["instant_bookable"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[52]:


#Logprice vs cleaning fee
plt.figure(figsize=(15,7))
sns.barplot(Airbnb1["cleaning_fee"],Airbnb1["log_price"],hue=Airbnb1["bathrooms"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[53]:


#Logprice vs accommodates
plt.figure(figsize=(15,7))
sns.pointplot(Airbnb1["accommodates"],Airbnb1["log_price"],hue=Airbnb1["cleaning_fee"])
plt.legend(bbox_to_anchor=(1.00, 1))
plt.show()


# In[54]:


plt.figure(figsize=(10,7))
sns.heatmap(Airbnb1.corr(),annot=True, cmap='gnuplot2')
plt.show;


# The highest correlation is seen in the relationship between the number of people a house or room can accomodates and number of beds it has. This is followed closely by number of bedrooms and number of beds available.
# 
# Log price is mostly correlated to the number of people a room or house can accomodate.
# 
# The least correlation we can see here is in the relationship of log price and number of bathrooms available
# 

# ### Linear Regression

# ### Model Building

# In[55]:


#Defining X and y variables
X = Airbnb.drop(['log_price'], axis=1)
y = Airbnb[['log_price']]

print(X.head())
print(y.head())


# In[56]:


print(X.shape)
print(y.shape)


# In[57]:


#split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### Preparing Data for Modelling
# 
# 
# #### One-hot encoding and Label Encoding

# In[ ]:





# In[58]:


df_dummies= pd.get_dummies(Airbnb, prefix='RoomType', columns=['room_type'])


# In[59]:


df_dummies


# In[60]:


df_dummies1= pd.get_dummies(Airbnb, prefix='CancelllationPolicy', columns=['cancellation_policy'])
df_dummies1


# In[61]:


df_dummies1.drop(columns=['room_type', 'accommodates', 'bathrooms', 'cleaning_fee', 'instant_bookable', 'bedrooms', 'beds', 'log_price'], inplace=True)
df_dummies1


# In[62]:


Airbnb1 = pd.concat([df_dummies,df_dummies1], axis=1)
Airbnb1.head()


# In[63]:


#dropping original cancellation policy column
Airbnb1.drop(columns=['cancellation_policy'], inplace=True)
Airbnb1.head(2)


# In[64]:


#replacing 'instant bookable' strings with numbers 0 and 1
Airbnb1['instant_bookable'] = Airbnb1['instant_bookable'].replace(['f', 't'], ['0', '1'])
Airbnb1.tail(2)


# In[65]:


#replacing 'cleaning fee' strings with numbers 0 and 1
Airbnb1['instant_bookable'] = Airbnb1['instant_bookable'].replace(['f', 't'], ['0', '1'])
Airbnb1.sample(2)


# In[66]:


from sklearn.preprocessing import LabelEncoder #import label encoder
labelencoder = LabelEncoder()
Airbnb2 = Airbnb1
Airbnb1['cleaning_fee'] = labelencoder.fit_transform(df_dummies.cleaning_fee) #returns label encoded variables
Airbnb2.sample(2)


# In[67]:


Airbnb2.isnull().sum()


# In[68]:


x = Airbnb2.drop('log_price', axis=1)
y = Airbnb2['log_price']


# In[69]:


# Splitting the data into train and test sets in 70:30 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1, shuffle=True)
x_train.shape, x_test.shape


# In[70]:


#Fitting linear model

from sklearn.linear_model import LinearRegression
linearregression = LinearRegression()                                    
linearregression.fit(x_train, y_train)                                  

print("Intercept of the linear equation:", linearregression.intercept_) 
print("\nCOefficients of the equation are:", linearregression.coef_)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = linearregression.predict(x_test)    


# In[71]:


# invoke the LinearRegression function and find the bestfit model on training data

regression_model = LinearRegression()
regression_model.fit(x_train, y_train)


# In[72]:


#get the R-square score the fitted train data

print('The coefficient of determination R^2 of the prediction on Train set', regression_model.score(x_train, y_train))


# In[73]:


# write your own R-square function for the testing data

def r_squared(model, x, y):
    y_mean = y_train.mean()
    SST = ((y_train - y_mean)**2).sum()
    SSE = ((y_train - regression_model.predict(x_train))**2).sum()
    r_square = 1 - SSE/SST
    return SSE, SST, r_square
    
SSE, SST, r_square = r_squared(regression_model, x_train, y_train)
print("SSE: ", SSE)
print("SST: ", SST)
print("r_square: ", r_square)


# #### Getting score on Test Set

# In[74]:


print('The coefficient of determination R^2 of the prediction on Test set',regression_model.score(x_test, y_test))


# In[75]:


print("The Root Mean Square Error (RMSE) of the model is for testing set is",np.sqrt(mean_squared_error(y_test,regression_model.predict(x_test))))


# In[76]:


a = regression_model.coef_
coeff_data = pd.DataFrame()
coeff_data['Coefs'] = regression_model.coef_
coeff_data['Feature'] = x_train.columns
coeff_data = coeff_data.append({'Coefs': regression_model.intercept_, 'Feature': "Intercept"}, ignore_index = True)
coeff_data


# In[77]:


# Let us write the equation of the fit
Equation = "log_price ="
print(Equation, end='\t')
for i in range(0, 13):
    if(i!=12):
        print("(",coeff_data.iloc[i].Coefs,")", "*", coeff_data.iloc[i].Feature, "+", end = '  ')
    else:
        print(coeff_data.iloc[i].Coefs)


# ### Using Stats Model OLS

# In[78]:


# This adds the constant term beta0 to the Linear Regression.
X_con=sm.add_constant(x)
x_trainc, x_testc, y_trainc, y_testc = train_test_split(X_con, y, test_size=0.30 , random_state=1)


# In[83]:


x_trainc=x_train.astype(int)


# In[84]:


model = sm.OLS(y_trainc,x_trainc).fit()
model.summary()


# In[85]:


print('The variation in the independent variable which is explained by the dependent variable is','\n',
      model.rsquared*100,'%')


# In[88]:


x_testc=x_test.astype(float)
ypred = model.predict(x_testc)


# In[89]:


mse = model.mse_model
mse


# ### Adding Interaction terms

# In[90]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train2 = poly.fit_transform(x_train)
x_test2 = poly.fit_transform(x_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(x_train2, y_train)

y_pred = poly_clf.predict(x_test2)

#print(y_pred)

#In sample (training) R^2 will always improve with the number of variables!
print(poly_clf.score(x_train2, y_train))


# In[91]:


#Out of sample (testing) R^2 is our measure of success
print(poly_clf.score(x_test2, y_test))


# In[92]:


# but this improves at the cost of 67 extra variables!
print(x_train.shape)
print(x_train2.shape)


# Polynomial Features (with only interaction terms) made the Out of sample R^2 worse even at the cost of a significant increase to the number of variables
# 
# 

# In[ ]:




