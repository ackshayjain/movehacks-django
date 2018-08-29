
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from time import time
from IPython.display import display
from matplotlib import pyplot as plt
data = pd.read_csv("data.csv")
display(data.head(n=5))


# In[2]:


# Working with the target variable
y = data['Total']
display(y.head(n=6))


# In[3]:


y.isnull().values.any()


# In[4]:


X = data.drop(['Invoice No', 'User', 'Out Time', 'In Date', 'Out Date', 'Total', 'MGP-MGN'], axis = 1)
display(X.head(n = 5))
X['MGN-WBT'] = X['MGN-WBT'].fillna("0")
X.shape
#X["LDG/UNLDG-WBG"].isnull().values.any()


# In[5]:


X['MGN-WBT'].isnull().values.any()


# In[6]:


X = X.drop(X.index[-1])
y = y.drop(y.index[-1])
def g(x):
    if x.startswith(":"):
        return x.replace(":", "0:")
    else:
        return x

X['MGN-WBT'] = X['MGN-WBT'].apply(lambda x: g(str(x)))
X['WBT-DOC'] = X['WBT-DOC'].apply(lambda x: g(str(x)))
X['DOC-LDG/UNLDG'] = X['DOC-LDG/UNLDG'].apply(lambda x: g(str(x)))
X['LDG/UNLDG-WBG'] = X['LDG/UNLDG-WBG'].apply(lambda x: g(str(x)))
X['WBG-INV'] = X['WBG-INV'].apply(lambda x: g(str(x)))
X['INV-MGX'] = X['INV-MGX'].apply(lambda x: g(str(x)))
display(X.head(n=5))
X.shape


# In[7]:


X['Truck No'] = X['Truck No'].apply(lambda x: x[:2] )
X['In Time'] = X["In Time"].apply(lambda x: x.split(':'))
#X['MGP-MGN'] = X["MGP-MGN"].apply(lambda x: x.split(':'))
X['MGN-WBT'] = X['WBT-DOC'].apply(lambda x: x.split(':'))
X['WBT-DOC'] = X['WBT-DOC'].apply(lambda x: x.split(':'))
X['DOC-LDG/UNLDG'] = X['DOC-LDG/UNLDG'].apply(lambda x: x.split(':'))
X['LDG/UNLDG-WBG'] = X['LDG/UNLDG-WBG'].apply(lambda x: x.split(':'))
X['WBG-INV'] = X['WBG-INV'].apply(lambda x: x.split(':'))
X['INV-MGX'] = X['INV-MGX'].apply(lambda x: x.split(':'))


display(X.head(n=5))


# In[8]:


X["In Time"] = X["In Time"].apply(lambda x: int(int(x[0])*60 + int(x[1])))
display(X.head(n=5))


# In[9]:


def k(x):
    if x[0] != 'nan':
        if x[0] == 0:
            return int(x[1])
        else:
            return int(x[0])*60 + int(x[1])       
    else:
        return None

X["MGN-WBT"] = X["MGN-WBT"].apply(lambda x: k(x))
X["WBT-DOC"] = X["WBT-DOC"].apply(lambda x: k(x))
X["DOC-LDG/UNLDG"] = X["DOC-LDG/UNLDG"].apply(lambda x: k(x))
X["LDG/UNLDG-WBG"] = X["LDG/UNLDG-WBG"].apply(lambda x: k(x))
X["WBG-INV"] = X["WBG-INV"].apply(lambda x: k(x))
X["INV-MGX"] = X["INV-MGX"].apply(lambda x: k(x))


# In[10]:


X.head(5)


# In[11]:


mean_MGN_WBT = X['MGN-WBT'].mean(skipna=True)
mean_WBT_DOC = X['MGN-WBT'].mean(skipna=True)
mean_DOC_LDG_UNLDG = X['MGN-WBT'].mean(skipna=True)
mean_LDG_UNLDG_WBG = X['MGN-WBT'].mean(skipna=True)
mean_WBG_INV = X['MGN-WBT'].mean(skipna=True)
mean_INV_MGX = X['MGN-WBT'].mean(skipna=True)


# In[12]:


X["MGN-WBT"] = X["MGN-WBT"].fillna(mean_MGN_WBT)
X["WBT-DOC"] = X["WBT-DOC"].fillna(mean_MGN_WBT)
X["DOC-LDG/UNLDG"] = X["DOC-LDG/UNLDG"].fillna(mean_MGN_WBT)
X["WBG-INV"] = X["WBG-INV"].fillna(mean_MGN_WBT)
X["INV-MGX"] = X["INV-MGX"].fillna(mean_MGN_WBT)
X["LDG/UNLDG-WBG"] = X["LDG/UNLDG-WBG"].fillna(mean_MGN_WBT)
display(X.head(n=5))


# In[13]:


y= y.apply(lambda x: x.split(':'))
def j(x):
   if x[0] != 'nan' and x[0] != '' :
       if x[0] == 0:
           return int(x[1])
       else:
           return int(x[0])*60 + int(x[1])       
   else:
       return None
y = y.apply(lambda x: j(x))
y.head(5)


# In[14]:


X_new = X.drop(['Truck No', 'Gate Entry No'], axis = 1)


# In[15]:


from sklearn.decomposition import PCA
X_final = np.array(X_new)
y_final = np.array(y)


# In[16]:


pca = PCA()
pca.fit(X_final)


# In[17]:


var_ratio = pca.explained_variance_ratio_
print(var_ratio)


# In[18]:


#labels = ['In Time', 'MGN_WBT', 'WBT-DOC', 'DOC-LDG/UNLDG', 'LDG/UNLDG-WBG', 'WBG-INV', "INV-MGX"]
sizes = var_ratio
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels = np.arange(7))
plt.show()


# In[19]:


plt.scatter(X['WBG-INV'], np.arange(X.shape[0]))
plt.xlabel("WBG_INV")
plt.ylabel("Samples")


# In[20]:


def pca_results(good_data, pca):
	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)


# In[21]:


pca_results = pca_results(X_new, pca)


# In[22]:


truck_no_unique = X["Truck No"].value_counts()
print(truck_no_unique)
truck_no_unique.shape


# In[23]:


plt.hist(truck_no_unique)


# In[24]:


def h(x):
    x = int(x)
    if x>0 and x<360:
        return 1
    if x>=360 and x<720:
        return 2
    if x>=720 and x<1080:
        return 3
    if x>=1080 and x<1440:
        return 4
X_new.head()


# In[25]:


X_new_zones = X_new
X_new_zones["In Time"] = X_new["In Time"].apply(lambda x: h(x))


# In[26]:


X_new_zones.head()


# In[27]:


X_new_zones_1 = X_new_zones
X_new_zones_1 = X_new_zones_1.where(X_new_zones_1["In Time"]==1)
X_new_zones_1 = X_new_zones_1.dropna()
X_new_zones_1 = X_new_zones_1.drop("In Time", axis = 1)
X_new_zones_1.head()


# In[28]:


X_new_zones_2 = X_new_zones
X_new_zones_2 = X_new_zones_2.where(X_new_zones_2["In Time"]==2)
X_new_zones_2 = X_new_zones_2.dropna()
X_new_zones_2 = X_new_zones_2.drop("In Time", axis = 1)
X_new_zones_2.shape


# In[29]:


X_new_zones_3 = X_new_zones
X_new_zones_3 = X_new_zones_3.where(X_new_zones_3["In Time"]==3)
X_new_zones_3 = X_new_zones_3.dropna()
X_new_zones_3 = X_new_zones_3.drop("In Time", axis = 1)
X_new_zones_3.shape


# In[30]:


X_new_zones_4 = X_new_zones
X_new_zones_4 = X_new_zones_4.where(X_new_zones_4["In Time"]==4)
X_new_zones_4 = X_new_zones_4.dropna()
X_new_zones_4 = X_new_zones_4.drop("In Time", axis = 1)
X_new_zones_4.shape


# In[31]:


pca_zone_1 = PCA()
X_zone_1_arr = np.array(X_new_zones_1)
pca_zone_1.fit(X_zone_1_arr)

pca_zone_2 = PCA()
X_zone_2_arr = np.array(X_new_zones_2)
pca_zone_2.fit(X_zone_2_arr)

pca_zone_3 = PCA()
X_zone_3_arr = np.array(X_new_zones_3)
pca_zone_3.fit(X_zone_3_arr)

pca_zone_4 = PCA()
X_zone_4_arr = np.array(X_new_zones_4)
pca_zone_4.fit(X_zone_4_arr)


# In[32]:


def pca_results(good_data, pca):
	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)


# In[33]:


pca_results(X_new_zones_1, pca_zone_1)


# In[34]:


pca_results(X_new_zones_2, pca_zone_2)


# In[35]:


pca_results(X_new_zones_3, pca_zone_3)


# In[36]:


pca_results(X_new_zones_4, pca_zone_4)


# In[37]:


X.head()
X_with_date = X
X_with_date["In Date"] = data['In Date']
X_with_date.head()


# In[38]:


y_with_date= X_with_date
y_with_date = y_with_date["In Date"]
y_with_date.head()
y_with_date_new = y_with_date.value_counts()
y_with_date_new


# In[39]:


y_with_date_new = X_with_date.drop(['Gate Entry No', 'Truck No', 'In Time', 'MGN-WBT', 'WBT-DOC', 'DOC-LDG/UNLDG', 'LDG/UNLDG-WBG','WBG-INV', 'INV-MGX'], axis = 1)

y_with_date_new['In Date'].value_counts()
y_un_dates = y_with_date_new['In Date'].unique()
s1 = pd.Series(y_un_dates, name = 'Date')
s2 = [162, 180, 147, 132, 125, 142, 146, 151, 162, 82, 108, 160, 136, 147, 129]
data_date = pd.DataFrame({'Date' :  s1, 'Trucks' : s2})

data_date


# In[40]:


import datetime
datetime.date(year = 2018, month = 6, day = 1).weekday()


# In[41]:


def fx(x):
    x = x.split('-')
    return datetime.date(year = int(x[2]), month = int(x[1]), day = int(x[0])).weekday()

def fg(x):
    x = x.split('-')
    return x[1]
data_date["Day"] = data_date["Date"].apply(lambda x: fx(x))
data_date['Month'] = data_date['Date'].apply(lambda x: fg(x))
data_date


# In[42]:


y_with_date = data_date['Trucks']


# In[43]:


X_date_final = data_date
X_date_final = X_date_final.drop(['Trucks', 'Date'], axis = 1)
X_date_final


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(X_date_final, y_with_date, test_size=0.2, random_state=42)


# In[45]:


reg = DecisionTreeRegressor(max_depth = 3)
reg.fit(X_train, y_train)


# In[46]:


pred_1 = reg.predict(X_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred_1)


# In[47]:


from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor()
reg2.fit(X_train, y_train)
pred2 = reg2.predict(X_test)
mean_absolute_error(y_test, pred2)
print (y_test)
print (pred2)

