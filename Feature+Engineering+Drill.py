
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set the default plot aesthetics to be prettier.
sns.set_style("white")


# In[20]:


df= pd.read_csv("ESSdata_Thinkful.csv", low_memory=False)
df


# In[21]:


print(df.head())


# In[23]:


features = pd.get_dummies(df['cntry'])


# # TLADS Drill

# 
# Categorize each of the variables in the ESS dataset as categorical or continuous, 
# and if continuous as ordinal, interval, or ratio variables. 
# Check your work with your mentor, and discuss what that information might 
# imply for feature engineering with this data.

# In[31]:


df.gndr.unique()


# In[33]:


df.dtypes


# https://thinkful-ed.github.io/data-201-resources/ESS_practice_data/ESS_codebook.html

# Country (cntry ): Nominal 
# 
# Respondent's identification number(idno ): Nominal	
# 
# ESS round (year): ordinal/nominal/ mostly Continuous
# 
# TV watching, total time on average weekday (tvtot): Continuous or discrete.
# 
# Most people can be trusted or you can't be too careful (ppltrst): Ordinal	
# 
# Most people try to take advantage of you, or try to be fair (pplfair): Nominal / Categorical
# 
# Most of the time people helpful or mostly looking out for themselves (pplhlp ): ?
# 
# How happy are you (happy): Ordinal
# 
# How often socially meet with friends, relatives or colleagues (sclmeet): Ratio
# 
# Take part in social activities compared to others of same age (sclact): Continuous
# 
# Gender (gndr): Nominal
# 
# Age of respondent, calculated (agea): Continuous Ratio
# 
# Lives with husband/wife/partner at household grid (partner): Nominal.

# #### Thinkful LESSON

# In[ ]:


features['Adult_65plus'] = np.where(df['agea']>=65, 1, 0)
print(df['agea'].groupby(features['Adult_65plus']).describe())


# In[ ]:


features['Nordic'] = np.where((df['cntry'].isin(['NO', 'SE'])), 1, 0)
print(pd.crosstab(features['Nordic'], df['cntry']))


# In[ ]:


corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

means = df[['ppltrst','pplfair','pplhlp']].mean(axis=0)
stds = df[['ppltrst','pplfair','pplhlp']].std(axis=0)
features['trust_fair_help'] = ((df[['ppltrst','pplfair','pplhlp']] - means) / stds).mean(axis=1)

plotdf = df.loc[:, ['ppltrst', 'pplfair', 'pplhlp']]
plotdf['trust_fair_help'] = features['trust_fair_help'] 
corrmat2 = plotdf.corr()

print(corrmat2)


# In[ ]:


#How do we know dist is not normal?

fig = plt.figure()

fig.add_subplot(221)
plt.hist(df['agea'].dropna())
plt.title('Raw')

fig.add_subplot(222)
plt.hist(np.log(df['agea'].dropna()))
plt.title('Log')

fig.add_subplot(223)
plt.hist(np.sqrt(df['agea'].dropna()))
plt.title('Square root')

ax3=fig.add_subplot(224)
plt.hist(1/df['agea'].dropna())
plt.title('Inverse')
plt.show()

features['log_age'] = np.log(df['agea'])



# In[ ]:


#Linear Relationships
sns.regplot(
    df['agea'],
    y=df['sclmeet'],
    y_jitter=.49,
    order=2,
    scatter_kws={'alpha':0.3},
    line_kws={'color':'black'},
    ci=None
)
plt.show()

features['age_squared'] = df['agea'] * df['agea']


# In[ ]:


features['Sadness'] = max(df['happy']) - df['happy']

sns.regplot(
    df['tvtot'],
    features['Sadness'],
    x_jitter=.49,
    y_jitter=.49,
    scatter_kws={'alpha':0.3},
    line_kws={'color':'black'},
    ci=None
)
plt.xlabel('TV watching')
plt.ylabel('Sadness')
plt.show()


# In[ ]:


from sklearn import preprocessing
df_num = df.select_dtypes(include=[np.number]).dropna()
names=df_num.columns
df_scaled = pd.DataFrame(preprocessing.scale(df_num), columns=names)
plt.scatter(df_num['tvtot'], df_scaled['tvtot'])
plt.show()
print(df_scaled.describe())




# In[ ]:



features['LiveWithPartner'] = np.where(df['partner'] == 1, 1, 0)


features['Sad_Partner'] = features['Sadness'] * features['LiveWithPartner']


features['tvtot'] = df['tvtot']
sns.lmplot(
    x='Sadness',
    y='tvtot',
    hue='LiveWithPartner',
    data=features,
    scatter=False
)
plt.show()

