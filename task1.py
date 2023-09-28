import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, linregress
from sklearn.preprocessing import StandardScaler

pd.set_option('chained_assignment', None)

MEDICAL_CLEAN = r'C:\Users\user\OneDrive\Documents\Education\Western Govenors University\MS - Data Analytics\D207 - ' \
                r'Exploratory Data Analysis\medical_clean.csv'

# Import the data
medical_clean = pd.read_csv(MEDICAL_CLEAN)


# A3 Data Identification
data = medical_clean[['ReAdmis', 'Initial_days']]
print(f'The data contains {data.shape[1]} columns and {data.shape[0]} observations.')
input('Enter to continue...\n')

# B1: Code

# View Initial_days distribution
fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle('Initial_days Distribution')
ax.hist(data.Initial_days, bins=int(max(data.Initial_days) / 10))
ax.set_xlabel('Initial_days')
ax.set_ylabel('Frequency')
plt.savefig('./output/initial_days_distribution.png')
print('Plot saved. Close the plot window to continue...')
plt.show()
plt.close()

# Assess variance by ReAdmis groups
a = data.query('ReAdmis == "Yes"').Initial_days
b = data.query('ReAdmis == "No"').Initial_days
print(f'The variance of Group A is: {a.var()}\n'
      f'The variance of Group B is: {b.var()}')
input('Enter to continue...\n')

# View transformed distribution
data['Initial_days_transformed'] = abs(data.Initial_days - data.Initial_days.mean())
fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle('Transformed Initial_days Distribution')
ax.hist(data.Initial_days_transformed)
ax.set_xlabel('abs(Initial_days - mean(Initial_days))')
ax.set_ylabel('Frequency')
plt.savefig('./output/initial_days_distribution_transformed.png')
print('Plot saved. Close the plot window to continue...\n')
plt.show()
plt.close()

# Assess variance by ReAdmis groups after transformation
a_transformed = data.query('ReAdmis == "Yes"').Initial_days_transformed
b_transformed = data.query('ReAdmis == "No"').Initial_days_transformed
print(f'The variance of Group A is: {a_transformed.var()}\n'
      f'The variance of Group B is: {b_transformed.var()}')
input('Enter to continue...\n')

# View group distributions
fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle('Transformed Initial_days Distribution by ReAdmis')
ax.hist(a_transformed, color='r', alpha=0.5, label='Yes')
ax.hist(b_transformed, color='b', alpha=0.5, label='No')
ax.set_xlabel('abs(Initial_days - mean(Initial_days))')
ax.set_ylabel('Frequency')
ax.legend(title='ReAdmis')
plt.savefig('./output/initial_days_distribution_by_readmis_transformed.png')
print('Plot Saved. Close the plot window to continue...\n')
plt.show()
plt.close()

# two-sample t-test right-tailed
result = ttest_ind(a_transformed, b_transformed, alternative='greater')
print(f'two-sample right-tailed t-test\n'
      f'\tt-statistic: {result.statistic}\n'
      f'\tp-value: {result.pvalue}')
input('Enter to continue...\n')

# Boxplots
fig, ax = plt.subplots()
fig.suptitle('Transformed Initial_days by ReAdmis')
ax.boxplot([a_transformed, b_transformed], labels=['Yes', 'No'], sym='')
ax.set_xlabel('ReAdmis')
ax.set_ylabel('abs(Initial_days - mean(Initial_days))')
print('Close the plot window to continue the program...\n')
plt.show()
plt.close()


# C: Univariate Statistics
print('Univariate Statistics\nCategorical Variables')
print(medical_clean[['Initial_admin', 'Services']].describe())
print(f'Unique Initial_admin Values:\n{medical_clean.Initial_admin.unique()}')
print(f'Unique Services Values:\n{medical_clean.Services.unique()}')
input('Enter to continue...\n')
print('Continuous Variables')
print(medical_clean[['Initial_days', 'TotalCharge']].describe())
print(f'Skewness:\n{medical_clean[["Initial_days", "TotalCharge"]].skew().to_string()}')
print(f'Kurtosis:\n{medical_clean[["Initial_days", "TotalCharge"]].kurt()}')
input('Enter to continue...\n')

# C1: Visual of Findings
# Add a count column
medical_clean['n'] = 1
# Prep Categorical Variables
cat1 = medical_clean[['Initial_admin', 'n']].groupby('Initial_admin', as_index=False).sum()
cat1.replace({'Elective Admission': 'Elective',
              'Emergency Admission': 'Emergency',
              'Observation Admission': 'Observation'}, inplace=True)
cat2 = medical_clean[['Services', 'n']].groupby('Services', as_index=False).sum()
cat2.replace({'Blood Work': 'Blood', 'CT Scan': 'CT', 'Intravenous': 'IV'}, inplace=True)
# Plots
fig, ax = plt.subplots(3, 2)
fig.set_tight_layout(True)
fig.suptitle('Univariate Statistics')
ax[0, 0].boxplot(medical_clean.Initial_days, vert=False)
ax[0, 0].set_title('Initial_days boxplot')
ax[0, 1].boxplot(medical_clean.TotalCharge, vert=False)
ax[0, 1].set_title('TotalCharge boxplot')
ax[1, 0].hist(medical_clean.Initial_days)
ax[1, 0].set_title('Initial_days histogram')
ax[1, 1].hist(medical_clean.TotalCharge)
ax[1, 1].set_title('TotalCharge histogram')
ax[2, 0].bar(cat1.Initial_admin, cat1.n)
ax[2, 0].set_title('Initial_admin barchart')
ax[2, 1].bar(cat2.Services, cat2.n)
ax[2, 1].set_title('Services barchart')
print('Close the plot window to continue the program...\n')
plt.show()
plt.close()

# D: Bivariate Statistics
# Categorical - Frequency Table
data_cat = medical_clean[['Initial_admin', 'Services']]
freq_table = pd.crosstab(data_cat.Services, data_cat.Initial_admin, margins=True)
print(f'Initial_admin vs Services Frequency Table:\n{freq_table.to_string()}')
input('Enter to continue...\n')

# Continuous - Correlation and R-Squared
data_con = medical_clean[['Initial_days', 'TotalCharge']]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_con)
x = data_scaled[:, 0]
y = data_scaled[:, 1]
results = linregress(x, y)
print('Initial_days vs TotalCharge Correlation and R-Squared')
print(f'Correlation Coefficient: {results.slope:}')
print(f'R-squared: {results.rvalue**2:}')
print(f'Line of best fit: y = {results.slope} * x + {results.intercept}')
input('Enter to continue...\n')

# D1: Visual of Findings

# Two Categorical Variables Distribution
# Add patient count column and group
data_cat['pat_count'] = 1
data_grp = data_cat.groupby(['Services', 'Initial_admin'], as_index=False).sum()
# Prepare plot variables
labels = [x for x in data_grp.Services.unique()]
emergency = data_grp.query('Initial_admin == "Emergency Admission"').pat_count
elective = data_grp.query('Initial_admin == "Elective Admission"').pat_count
observation = data_grp.query('Initial_admin == "Observation Admission"').pat_count
x = np.arange(len(labels))  # bar locations
width = 0.3    # bar widths
# Create plot
fig, ax = plt.subplots()
fig.set_tight_layout(True)
fig.suptitle('pat_count by Services and Initial_admin'),
bar1 = ax.bar(x, emergency, width, label='Emergency Admission')
bar2 = ax.bar(x + width, elective, width, label='Elective Admission')
bar3 = ax.bar(x + (2 * width), observation, width, label='Observation Admission')
# Label plot
ax.set_ylabel('pat_count')
ax.set_xlabel('Services')
ax.set_xticks(x + width, labels)
ax.legend(title='Initial_admin')
ax.bar_label(bar1)
ax.bar_label(bar2)
ax.bar_label(bar3)
print('Close the plot window to continue...\n')
plt.show()
plt.close()

# Two Continuous Variables Distribution
x = data_scaled[:, 0]
y = data_scaled[:, 1]
fig, ax = plt.subplots()
fig.suptitle('Initial_days vs TotalCharge')
ax.scatter(x, y, marker='.', alpha=0.1, label='distribution')
ax.set_xlabel('Initial_days (scaled)')
ax.set_ylabel('TotalCharge (scaled)')
plt.plot(x, results.slope * x + results.intercept, 'black', label='fitted line')
plt.legend()
print('Close the plot window to continue the program...')
plt.show()
plt.close()
