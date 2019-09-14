
# coding: utf-8

# ## Olanipekun Ayorinde 

# In[35]:


import pandas as pd 

import matplotlib.pylab as plt 


# In[36]:


filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"


# In[37]:


headers = ["symboling","normalized-losses","makes","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]


# In[38]:


df=pd.read_csv(filename, names = headers)


# In[40]:


# to see the data set, we'll use the head() method
df.head()


# In[41]:


import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


# In[42]:


missing_data = df.isnull()
missing_data.head(5)


# In[43]:


# counting missing values in each column

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[44]:


# to calculate the average of the column

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses", avg_norm_loss)


# In[45]:


# Replace "NaN" by mean value in "normalized-losses" column

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# In[46]:


# calculate the mean value for 'bore' column

avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[47]:


# Replace NaN by mean value

df["bore"].replace(np.nan, avg_bore, inplace=True)


# In[48]:


# Calculate the mean value for "stroke" column

avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print ("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# In[49]:


# Calculate the mean value for the 'horsepower' column:

avg_horsepower = df['horsepower'].astype('float').mean(axis=0) 
print("Average horsepower:", avg_horsepower)


# In[50]:


# Replace "NaN" by mean value 

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# In[51]:


# Calculate the mean value for 'peak-rpm' column

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# In[52]:


# Replace NaN by mean value

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# In[53]:


# To see which value are presnt in each column 

df['num-of-doors'].value_counts()


# In[54]:


# To calculate the most common type automatically

df['num-of-doors'].value_counts().idxmax()


# In[55]:


# replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace (np.nan, "four", inplace=True)


# In[56]:


# to drop all rows that did not have price data
#simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we dropped two rows
df.reset_index(drop=True, inplace=True)


# In[57]:


# To obtain dataset with no missing values

df.head()


# In[58]:


# Last step is to check if all data are in correct data format 

df.dtypes


# In[59]:


# Coverting data to proper format 
df[["bore","stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-arm"]] = df[["peak-rpm"]].astype("float")
                                                       
   


# In[60]:


# To finally have a cleaned datasets 
df.dtypes


# In[61]:


# we finally have a cleaned datasets with no missing values and all data in its proper formats 


# In[62]:


# Data standardization

df.head()


# In[63]:


# convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data
df.head()


# In[64]:


# According to the example above, tranform mpg to L/100km in the column of "highway-mpg", and change the name of column to "highway-L/100km"

# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)

# check your transformed data 
df.head


# In[65]:


# Normalization 
# Normalization is the process of transforming values of several variables into a similar range. 

# would like to normalize varibles so their value ranges from 0 to 1
# Approach: Replace original value by (original value)/(maximum value)

# replace (original value) by (original value)/(maximum value)

df['lenghth'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[66]:


# to normalize the column "height"

df['height'] = df['height']/df['height'].max()
# show the scaled columns
df[["length","width","height"]].head()


# In[67]:


# from above we have normalized "length","width", and "height" in the range of [0,1]


# In[68]:


# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis


# In[69]:


#  in our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values. what if we only care about about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3types)? can we rearrange them into three 'bins' to simplify analysis?


# In[71]:



# we will use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins

df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[70]:


# Lets plot the histogram of horsepower, to see what the distribution of horsepower looks like 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[72]:


# we would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated) function

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[73]:


group_names  = ['Low', 'Medium', 'High']


# In[74]:


# we apply the fnction "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower', 'horsepower-binned']].head(20)


# In[75]:


df["horsepower-binned"].value_counts()


# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[77]:


# Bins Visulaization
# Histogram is used to visualize the distribution of bins we created above 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt 
from matplotlib import pyplot

a = (0,1,2)

# draw histogram of attribute "horsepower" with bins =3
plt.pyplot.hist(df["horsepower"], bins = 3)

#set x/y labels and plot title 
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")







# In[78]:


# Indicator variable ( or dummy variable)


# In[79]:


df.columns


# In[80]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[81]:


# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original "fuel-type", axis =1, inplace=True)


# In[82]:


df.head()


# In[92]:


# get indicator varibales of aspiration and assign it to data frame "dummy_variable_2"
dummy_variables_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity 
dummy_variables_2.rename(columns={'std':'aspiration-std','turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instaces of data frame "dummy_variable_1"
dummy_variables_2.head()


# In[95]:


#merge the new dataframe to the original dataform 

df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# In[97]:


df.to_csv('clean_df.csv')

