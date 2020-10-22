#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

plt.style.use('ggplot')

# In[81]:


Log_TwoPoint_EG1 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_TwoPoint_EG1.csv',
                               header=0)
Log_TwoPoint_EG2 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_TwoPoint_EG2.csv',
                               header=0)

Log_TwoPoint_EG3 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_TwoPoint_EG3.csv',
                               header=0)

Log_Uniform_EG1 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_Uniform_EG1.csv',
                              header=0)
Log_Uniform_EG2 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_Uniform_EG2.csv',
                              header=0)

Log_Uniform_EG3 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_Logbook_Uniform_EG3.csv',
                              header=0)

# In[82]:


TEG1G = Log_TwoPoint_EG1['Generation']
TEG1AA = Log_TwoPoint_EG1['avg_avg']
TEG1AM = Log_TwoPoint_EG1['avg_max']
TEG1SA = Log_TwoPoint_EG1['std_avg']
TEG1SM = Log_TwoPoint_EG1['std_max']

TEG2G = Log_TwoPoint_EG2['Generation']
TEG2AA = Log_TwoPoint_EG2['avg_avg']
TEG2AM = Log_TwoPoint_EG2['avg_max']
TEG2SA = Log_TwoPoint_EG2['std_avg']
TEG2SM = Log_TwoPoint_EG2['std_max']

TEG3G = Log_TwoPoint_EG3['Generation']
TEG3AA = Log_TwoPoint_EG3['avg_avg']
TEG3AM = Log_TwoPoint_EG3['avg_max']
TEG3SA = Log_TwoPoint_EG3['std_avg']
TEG3SM = Log_TwoPoint_EG3['std_max']
##
UEG1G = Log_Uniform_EG1['Generation']
UEG1AA = Log_Uniform_EG1['avg_avg']
UEG1AM = Log_Uniform_EG1['avg_max']
UEG1SA = Log_Uniform_EG1['std_avg']
UEG1SM = Log_Uniform_EG1['std_max']

UEG2G = Log_Uniform_EG2['Generation']
UEG2AA = Log_Uniform_EG2['avg_avg']
UEG2AM = Log_Uniform_EG2['avg_max']
UEG2SA = Log_Uniform_EG2['std_avg']
UEG2SM = Log_Uniform_EG2['std_max']

UEG3G = Log_Uniform_EG3['Generation']
UEG3AA = Log_Uniform_EG3['avg_avg']
UEG3AM = Log_Uniform_EG3['avg_max']
UEG3SA = Log_Uniform_EG3['std_avg']
UEG3SM = Log_Uniform_EG3['std_max']

# In[126]:


import math
import seaborn as sns

figure, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))

a = (1 / (math.sqrt(30)))
# a=1

axes[0].set_title('EG 1')
axes[1].set_title('EG 2')
axes[2].set_title('EG 3')

axes[0].set(ylabel='Fitness')

axes[0].set(xlabel='Generation')
axes[1].set(xlabel='Generation')
axes[2].set(xlabel='Generation')

axes[0].plot(UEG1G, UEG1AA, label='Uniform_Avg')
axes[0].plot(UEG1G, UEG1AM, label='Uniform_Max')
axes[0].fill_between(UEG1G, (UEG1AA - a * UEG1SA), (UEG1AA + a * UEG1SA), alpha=0.3)
axes[0].fill_between(UEG1G, (UEG1AM - a * UEG1SM), (UEG1AM + a * UEG1SM), alpha=0.3)

axes[0].plot(TEG1G, TEG1AA, label='TwoPoint_Avg')
axes[0].plot(TEG1G, TEG1AM, label='TwoPoint_Max')
axes[0].fill_between(TEG1G, (TEG1AA - a * TEG1SA), (TEG1AA + a * TEG1SA), alpha=0.3)
axes[0].fill_between(TEG1G, (TEG1AM - a * TEG1SM), (TEG1AM + a * TEG1SM), alpha=0.3)

axes[1].plot(UEG2G, UEG2AA, label='Uniform_Avg')
axes[1].plot(UEG2G, UEG2AM, label='Uniform_Max')
axes[1].fill_between(UEG2G, (UEG2AA - a * UEG2SA), (UEG2AA + a * UEG2SA), alpha=0.3)
axes[1].fill_between(UEG2G, (UEG2AM - a * UEG2SM), (UEG2AM + a * UEG2SM), alpha=0.3)

axes[1].plot(TEG2G, TEG2AA, label='TwoPoint_Avg')
axes[1].plot(TEG2G, TEG2AM, label='TwoPoint_Max')
axes[1].fill_between(TEG2G, (TEG2AA - a * TEG2SA), (TEG2AA + a * TEG2SA), alpha=0.3)
axes[1].fill_between(TEG2G, (TEG2AM - a * TEG2SM), (TEG2AM + a * TEG2SM), alpha=0.3)

axes[2].plot(UEG3G, UEG3AA, label='Uniform_Avg')
axes[2].plot(UEG3G, UEG3AM, label='Uniform_Max')
axes[2].fill_between(UEG3G, (UEG3AA - a * UEG3SA), (UEG3AA + a * UEG3SA), alpha=0.3)
axes[2].fill_between(UEG2G, (UEG3AM - a * UEG3SM), (UEG3AM + a * UEG3SM), alpha=0.3)

axes[2].plot(TEG3G, TEG3AA, label='TwoPoint_Avg')
axes[2].plot(TEG3G, TEG3AM, label='TwoPoint_Max')
axes[2].fill_between(TEG3G, (TEG3AA - a * TEG3SA), (TEG3AA + a * TEG3SA), alpha=0.3)
axes[2].fill_between(TEG3G, (TEG3AM - a * TEG3SM), (TEG3AM + a * TEG3SM), alpha=0.3)

axes[2].legend(loc="lower right")

# In[143]:


IG_TwoPoint_EG1 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_TwoPoint_EG1.csv',
                              header=0)

IG_TwoPoint_EG2 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_TwoPoint_EG2.csv',
                              header=0)
IG_TwoPoint_EG3 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_TwoPoint_EG3.csv',
                              header=0)

IG_Uniform_EG1 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_Uniform_EG1.csv',
                             header=0)

IG_Uniform_EG2 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_Uniform_EG2.csv',
                             header=0)

IG_Uniform_EG3 = pd.read_csv(r'C:\Users\user\Desktop\Task2\General_IGs_Uniform_EG3.csv',
                             header=0)

data = [IG_TwoPoint_EG1['best_ig'], IG_Uniform_EG1['best_ig'], IG_TwoPoint_EG2['best_ig'],
        IG_Uniform_EG2['best_ig'], IG_TwoPoint_EG3['best_ig'], IG_Uniform_EG3['best_ig']]

fig = plt.figure(figsize=(19, 5))
ax = fig.add_subplot(111)

plt.boxplot(data)

plt.xticks([1, 2, 3, 4, 5, 6],
           ['TwoPoint_EG1', 'Uniform_EG1', 'TwoPoint_EG2', 'Uniform_EG2', 'TwoPoint_EG3', 'Uniform_EG3'], fontsize=15)
plt.yticks(size=15)

plt.ylabel('Individual Gain', fontsize=15)
plt.title('Boxplots Individual Gain', fontsize=15)

# In[105]:


# Compare between TwoPoint and Uniform
print('Results for comparing Averages TwoPoint and Uniform, EG1:')
print(mannwhitneyu(TEG1AA, UEG1AA))

print('Results for comparing MAX TwoPoint and Uniform, EG1:')
print(mannwhitneyu(TEG1AM, UEG1AM))

print('Results for comparing Averages TwoPoint and Uniform, EG2:')
print(mannwhitneyu(TEG2AA, UEG2AA))

print('Results for comparing MAX TwoPoint and Uniform, EG2:')
print(mannwhitneyu(TEG2AM, UEG2AM))

print('Results for comparing Averages TwoPoint and Uniform, EG3:')
print(mannwhitneyu(TEG3AA, UEG3AA))

print('Results for comparing MAX TwoPoint and Uniform, EG3:')
print(mannwhitneyu(TEG3AM, UEG3AM))

# In[106]:


# Compare between EG's
print('Results for comparing Averages, TwoPoint, EG1 VS EG2:')
print(mannwhitneyu(TEG1AA, TEG2AA))

print('Results for comparing MAX, TwoPoint, EG1 VS EG2:')
print(mannwhitneyu(TEG1AM, TEG2AM))

print('Results for comparing Averages, TwoPoint, EG1 VS EG3:')
print(mannwhitneyu(TEG1AA, TEG3AA))

print('Results for comparing MAX, TwoPoint, EG1 VS EG3:')
print(mannwhitneyu(TEG1AM, TEG3AM))

print('Results for comparing Averages, TwoPoint, EG2 VS EG3:')
print(mannwhitneyu(TEG2AA, TEG3AA))

print('Results for comparing MAX, TwoPoint, EG2 VS EG3:')
print(mannwhitneyu(TEG2AM, TEG3AM))

#

print('Results for comparing Averages, Uniform, EG1 VS EG2:')
print(mannwhitneyu(UEG1AA, UEG2AA))

print('Results for comparing MAX, Uniform, EG1 VS EG2:')
print(mannwhitneyu(UEG1AM, UEG2AM))

print('Results for comparing Averages, Uniform, EG1 VS EG3:')
print(mannwhitneyu(UEG1AA, UEG3AA))

print('Results for comparing MAX, Uniform, EG1 VS EG3:')
print(mannwhitneyu(UEG1AM, UEG3AM))

print('Results for comparing Averages, Uniform, EG2 VS EG3:')
print(mannwhitneyu(UEG2AA, UEG3AA))

print('Results for comparing MAX, Uniform, EG2 VS EG3:')
print(mannwhitneyu(UEG2AM, UEG3AM))

# In[107]:


# Compare between TwoPoint and Uniform for boxplot
print('Results for comparing Boxplot IG, TwoPoint and Uniform, EG1:')
print(mannwhitneyu(IG_TwoPoint_EG1['best_ig'], IG_Uniform_EG1['best_ig']))

print('Results for comparing Boxplot IG, TwoPoint and Uniform, EG2:')
print(mannwhitneyu(IG_TwoPoint_EG2['best_ig'], IG_Uniform_EG2['best_ig']))

print('Results for comparing Boxplot IG, TwoPoint and Uniform, EG3:')
print(mannwhitneyu(IG_TwoPoint_EG3['best_ig'], IG_Uniform_EG3['best_ig']))

# In[108]:


# Compare between EG's
print('Results for comparing Boxplot IG, TwoPoint, EG1 VS EG2:')
print(mannwhitneyu(IG_TwoPoint_EG1['best_ig'], IG_TwoPoint_EG2['best_ig']))

print('Results for comparing Boxplot IG, TwoPoint, EG1 VS EG3:')
print(mannwhitneyu(IG_TwoPoint_EG1['best_ig'], IG_TwoPoint_EG3['best_ig']))

print('Results for comparing Boxplot IG, TwoPoint, EG2 VS EG3:')
print(mannwhitneyu(IG_TwoPoint_EG2['best_ig'], IG_TwoPoint_EG3['best_ig']))

print('Results for comparing Boxplot IG, Uniform, EG1 VS EG2:')
print(mannwhitneyu(IG_Uniform_EG1['best_ig'], IG_Uniform_EG2['best_ig']))

print('Results for comparing Boxplot IG, Uniform, EG1 VS EG3:')
print(mannwhitneyu(IG_Uniform_EG1['best_ig'], IG_Uniform_EG3['best_ig']))

print('Results for comparing Boxplot IG, Uniform, EG2 VS EG3:')
print(mannwhitneyu(IG_Uniform_EG2['best_ig'], IG_Uniform_EG3['best_ig']))

# In[116]:


print('Average, TwoPoint, EG1:')
print(TEG1AA.mean(), TEG1AA.std())

print('Average, TwoPoint, EG2:')
print(TEG2AA.mean(), TEG2AA.std())

print('Average, TwoPoint, EG3:')
print(TEG3AA.mean(), TEG3AA.std())

print('Max, TwoPoint, EG1:')
print(TEG1AM.mean(), TEG1AM.std())

print('Max, TwoPoint, EG2:')
print(TEG2AM.mean(), TEG2AM.std())

print('Max, TwoPoint, EG3:')
print(TEG3AM.mean(), TEG3AM.std())
#

print('Average, Uniform, EG1:')
print(UEG1AA.mean(), UEG1AA.std())

print('Average, Uniform, EG2:')
print(UEG2AA.mean(), UEG2AA.std())

print('Average, Uniform, EG3:')
print(UEG3AA.mean(), UEG3AA.std())

print('Max, Uniform, EG1:')
print(UEG1AM.mean(), UEG1AM.std())

print('Max, Uniform, EG2:')
print(UEG2AM.mean(), UEG2AM.std())

print('Max, Uniform, EG3:')
print(UEG3AM.mean(), UEG3AM.std())

# In[124]:


print('Boxplot IG, TwoPoint, EG1, mean and std')
print(IG_TwoPoint_EG1['best_ig'].mean(), IG_TwoPoint_EG1['best_ig'].std())

print('Boxplot IG, TwoPoint, EG2, mean and std')
print(IG_TwoPoint_EG2['best_ig'].mean(), IG_TwoPoint_EG2['best_ig'].std())

print('Boxplot IG, TwoPoint, EG3, mean and std')
print(IG_TwoPoint_EG3['best_ig'].mean(), IG_TwoPoint_EG3['best_ig'].std())

#

print('Boxplot IG, Uniform, EG1, mean and std')
print(IG_Uniform_EG1['best_ig'].mean(), IG_Uniform_EG1['best_ig'].std())

print('Boxplot IG, Uniform, EG2, mean and std')
print(IG_Uniform_EG2['best_ig'].mean(), IG_Uniform_EG2['best_ig'].std())

print('Boxplot IG, Uniform, EG3, mean and std')
print(IG_Uniform_EG3['best_ig'].mean(), IG_Uniform_EG3['best_ig'].std())

# In[ ]:





