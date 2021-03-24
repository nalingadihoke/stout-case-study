#!/usr/bin/env python
# coding: utf-8

# In[151]:


# import statements

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[152]:


# read dataframe
df1 = pd.read_csv('/Users/nalingadihoke/Desktop/Stout_Case_Study/Question_2/data/casestudy.csv')
df1 = df1.reset_index()


# In[153]:


df1 = df1.drop(['Unnamed: 0'], axis = 1)
df1.head()


# In[154]:


df2 = df1.copy()


# In[155]:


df1['customer_email'] = df1['customer_email'].str.rstrip().str.lstrip()


# In[156]:


size = df1.shape
rows  = size[0]
cols  = size[1]
print(rows, cols)


# In[157]:


years = df1.year.unique()
num_yrs = len(years)
print(years, num_yrs)


# In[183]:


#loop from here
y1 = 2016
y2 = 2015
current_year = df1['year'] == y1
current = df1[current_year]

if y1 == 2015:
    blank = {'index': [], 'customer_email': [], 'net_revenue': [], 'year': []}
    previous = pd.DataFrame(blank)
else:
    previous_year = df1['year'] == y2
    previous = df1[previous_year]

# 1. total customers current year
tot_curr = current.shape[0]

# 2. total customers previous year
tot_prev = previous.shape[0]

print('total customers current year: ', tot_curr)
print('total customers previous year: ', tot_prev)


# In[161]:


current.head()


# In[190]:


curr_customers = current['customer_email']
prev_customers = previous['customer_email']

len(set(prev_customers))


# In[185]:


# 3. Total revenue for current year

total_revenue_current = current['net_revenue'].sum()

print('total revenue for current year:', total_revenue_current)


# In[169]:


new_customers = set(curr_customers) - set(prev_customers)

existing_customers = set(curr_customers) & set(prev_customers)

print('new_customers:', len(new_customers))
print('existing_customers:', len(existing_customers))


# In[170]:


e_curr = current['customer_email'].isin(existing_customers)
existing_current_year = current[e_curr]

e_prev = previous['customer_email'].isin(existing_customers)
existing_previous_year = previous[e_prev]


# In[172]:


len(existing_current_year['customer_email'])


# In[173]:


n_curr = current['customer_email'].isin(new_customers)
new_current_year = current[n_curr]

n_prev = previous['customer_email'].isin(new_customers)
new_previous_year = previous[n_prev]


# In[176]:


len(new_current_year['index'])


# In[186]:


# 4. New customer revenue

total_revenue_new_customers_current = new_current_year['net_revenue'].sum()

print('new customer revenue (new customers not in previous year only):', total_revenue_new_customers_current)


# In[187]:


# 5. Existing customer growth = Revenue of existing customers for current year – Revenue of existing customers from previous year

total_revenue_existing_customers_current = existing_current_year['net_revenue'].sum()
total_revenue_existing_customers_previous = existing_previous_year['net_revenue'].sum()

existing_customer_growth = total_revenue_existing_customers_current - total_revenue_existing_customers_previous

print('existing customer growth:', existing_customer_growth)


# In[188]:


# 7. Existing customer revenue current year
# 8. Existing customer revenue previous year

print('existing customer current year:', total_revenue_existing_customers_current)
print('existing customer previous year:', total_revenue_existing_customers_previous)


# In[191]:


# 9. Number of new customers
# 10. Number of customers lost

new_customers = set(curr_customers) - set(prev_customers)

lost_customers = set(prev_customers) - set(curr_customers)

print('new_customers:', len(new_customers))
print('lost_customers:', len(lost_customers))


# In[192]:


l_curr = current['customer_email'].isin(lost_customers)
lost_current_year = current[l_curr]

l_prev = previous['customer_email'].isin(lost_customers)
lost_previous_year = previous[l_prev]


# In[195]:


len(lost_previous_year['index'])


# In[196]:


# 6. Revenue lost from attrition

total_revenue_lost_customers_previous = lost_previous_year['net_revenue'].sum()

print('revenue lost from attrition (revenue from customers lost from previous year):', total_revenue_lost_customers_previous)


# In[213]:


def value_calculator(y1, y2, df1):
    
    # create 2 dfs for current and previous year
    
    current_year = df1['year'] == y1
    current = df1[current_year]

    if y1 == 2015:
        blank = {'index': [], 'customer_email': [], 'net_revenue': [], 'year': []}
        previous = pd.DataFrame(blank)
    else:
        previous_year = df1['year'] == y2
        previous = df1[previous_year]
        
    # 1. total customers current year
    tot_curr = current.shape[0]

    # 2. total customers previous year
    tot_prev = previous.shape[0]

    #print('total customers current year: ', tot_curr)
    #print('total customers previous year: ', tot_prev)
    
    # 3. Total revenue for current year

    total_revenue_current = current['net_revenue'].sum()

    #print('total revenue for current year:', total_revenue_current)
    
    
    # New, existing and lost 
    
    curr_customers = current['customer_email']
    prev_customers = previous['customer_email']
    
    new_customers = set(curr_customers) - set(prev_customers)

    existing_customers = set(curr_customers) & set(prev_customers)
    
    lost_customers = set(prev_customers) - set(curr_customers)
    
    # 9. Number of new customers
    # 10. Number of customers lost

    #print('new_customers:', len(new_customers))
    #print('existing_customers:', len(existing_customers))
    #print('lost_customers:', len(lost_customers))
    
    
    e_curr = current['customer_email'].isin(existing_customers)
    existing_current_year = current[e_curr]

    e_prev = previous['customer_email'].isin(existing_customers)
    existing_previous_year = previous[e_prev]
    
    n_curr = current['customer_email'].isin(new_customers)
    new_current_year = current[n_curr]

    n_prev = previous['customer_email'].isin(new_customers)
    new_previous_year = previous[n_prev]
    
    l_curr = current['customer_email'].isin(lost_customers)
    lost_current_year = current[l_curr]

    l_prev = previous['customer_email'].isin(lost_customers)
    lost_previous_year = previous[l_prev]
    
    # 4. New customer revenue

    total_revenue_new_customers_current = new_current_year['net_revenue'].sum()

    #print('new customer revenue (new customers not in previous year only):', total_revenue_new_customers_current)
    
    
    # 5. Existing customer growth = Revenue of existing customers for current year – Revenue of existing customers from previous year

    total_revenue_existing_customers_current = existing_current_year['net_revenue'].sum()
    total_revenue_existing_customers_previous = existing_previous_year['net_revenue'].sum()

    existing_customer_growth = total_revenue_existing_customers_current - total_revenue_existing_customers_previous

    #print('existing customer growth:', existing_customer_growth)
    
    # 6. Revenue lost from attrition

    total_revenue_lost_customers_previous = lost_previous_year['net_revenue'].sum()
    
    #print('revenue lost from attrition (revenue from customers lost from previous year):', total_revenue_lost_customers_previous)
    
    # 7. Existing customer revenue current year
    # 8. Existing customer revenue previous year

    #print('existing customer revenue current year:', total_revenue_existing_customers_current)
    #print('existing customer revenue previous year:', total_revenue_existing_customers_previous)
    
    d = {'current_year': y1,
         'previous_year': y2,
         'total_customers_current_year': tot_curr,
         'total_customers_previous_year': tot_prev,
         'new_customers': len(new_customers),
         'lost_customers': len(lost_customers),
         'existing_customers': len(existing_customers),
         'existing_customer_revenue_current_year': total_revenue_existing_customers_current,
         'existing_customer_revenue_previous_year': total_revenue_existing_customers_previous,
         'existing_customer_revenue_growth': existing_customer_growth,
         'revenue_lost_from_attrition': total_revenue_lost_customers_previous,
         'total_revenue_current_year': total_revenue_current,
         'new_customer_revenue': total_revenue_new_customers_current
        }
    
    return d


# In[214]:


d1 = value_calculator(2015, 2014, df1.copy())
d2 = value_calculator(2016, 2015, df1.copy())
d3 = value_calculator(2017, 2016, df1.copy())


# In[215]:


d1


# In[216]:


ds = [d1, d2, d3]
d = {}
for k in d1.keys():
    d[k] = list(d[k] for d in ds)


# In[217]:


result = pd.DataFrame.from_dict(d)

result


# In[218]:


result.to_markdown()


# In[ ]:




