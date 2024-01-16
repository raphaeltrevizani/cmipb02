#!/usr/bin/python

import pandas as pd
from scipy.stats import spearmanr
import joblib
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------
def parse_files(plasma_file, specimen_file):

	# Parse plama file
	df_plasma = pd.read_csv(plasma_file, sep='\t')[['specimen_id', 'isotype', 'antigen', 'MFI_normalised']]

	# Parse subject file for IDs
	df_specimen = pd.read_csv(specimen_file, sep='\t')[['specimen_id', 'subject_id', 'planned_day_relative_to_boost']]

	# Combine specimen and plasma files
	merged_df = pd.merge(df_specimen, df_plasma, on='specimen_id')


	## X data:
	# Remove all days except for day == 0
	df_day0 = merged_df[merged_df['planned_day_relative_to_boost'] == 0]
	# print(df_day0.to_string())

	# This is always 0 from now on, so I'm removing it
	del df_day0['planned_day_relative_to_boost']

	# Combine isotype and antigen into one column
	df_day0['Isotype/Antigen'] = df_day0['isotype'].astype(str) + '/' + df_day0['antigen'].astype(str)
	del df_day0['isotype']
	del df_day0['antigen']

	df_day0 = df_day0.drop_duplicates(subset=['specimen_id', 'Isotype/Antigen'])
	
	## Y data:
	# Selects rows with the answer: 14-days, PT, IgG
	planned_day14 = merged_df['planned_day_relative_to_boost'] == 14
	isotype_IgG   = merged_df['isotype'] == 'IgG'
	total_antigen = merged_df['antigen'] == 'PT'
	df_day14 = merged_df[planned_day14 & isotype_IgG & total_antigen]
	
	# Removes unnecessary columns from answer dataset
	df_day14 = df_day14[['specimen_id','subject_id', 'MFI_normalised']]
	df_day14['Isotype/Antigen'] = 'IgG+14days'

	# # 
	# df = pd.merge(df_day0, df_day14, on='subject_id')
	df = pd.concat([df_day0, df_day14], ignore_index=True)

	# At this point, subject and specimen are the same so I'm removing specimen_id
	del df['specimen_id']

	return df

# ----------------------------------------------------------

plasma_file2020 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2020LD_plasma_ab_titer.tsv'
specimen_file2020 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2020LD_specimen.tsv'

plasma_file2021 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2021LD_plasma_ab_titer.tsv'
specimen_file2021 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2021LD_specimen.tsv'

plasma_file2022 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/prediction_data/2022BD_plasma_ab_titer.tsv'
specimen_file2022 = 'data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/prediction_data/2022BD_specimen.tsv'


df_2020 = parse_files(plasma_file2020, specimen_file2020)
df_2021 = parse_files(plasma_file2021, specimen_file2021)
df_2022 = parse_files(plasma_file2022, specimen_file2022)

# Concatenate 2020 and 2021
df_raw = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)

# Reshaping data 
df_xy = df_raw.pivot(index='subject_id', columns='Isotype/Antigen', values='MFI_normalised')

# Removing columns where data is missing
df_xy_temp = df_xy.dropna(axis=1)

subject_ids_2022 = list(set(df_2022['subject_id'].to_list()))

df_reshaped_2022 = df_xy_temp[df_xy_temp.index.isin(subject_ids_2022)]


# Removing rows where Y-data is missing
df_xy = df_xy.dropna(subset=['IgG+14days'])

# Removing columns where data is missing
df_xy = df_xy.dropna(axis=1)

common_columns = list(set(df_xy.columns).intersection(set(df_reshaped_2022.columns)))

df_xy_tmp = df_xy[common_columns]
df_xy = pd.merge(df_xy_tmp, df_xy['IgG+14days'], left_index=True, right_index=True)
df_reshaped_2022 = df_reshaped_2022[common_columns]
# Reordering to make target column last column
except_target = [x for x in list(df_xy.columns) if x != 'IgG+14days']

df = df_xy[except_target + ['IgG+14days']]

# Fold change
df['IgG+14days'] = df['IgG+14days']/df['IgG/PT']


array = df.values
X_train = array[:,:-1]
Y_train = array[:,-1]

# More models
models = []
models.append(('RF', RandomForestRegressor()))

for model_tuple in models:

	model_name = model_tuple[0]
	model_func = model_tuple[1]
	model_func.fit(X_train, Y_train)
	
model = models[0]

# 1B. Test model against the 2021 data
X_pred = df_reshaped_2022.values
# X_test = array[:,:-1]

model_name = model[0]
model_func = model[1]
Y_pred = model_func.predict(X_pred)

print(Y_pred)

df_reshaped_2022['task1.2'] = Y_pred
df_reshaped_2022 = df_reshaped_2022.sort_values('task1.2', ascending=False)
print(df_reshaped_2022)

df_reshaped_2022['task1.2'].to_csv('output/task/task1.2_predictions.csv')