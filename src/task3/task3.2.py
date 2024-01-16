#!/usr/bin/python

import pandas as pd
from scipy.stats import spearmanr
import joblib
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------
def parse_files(plasma_file, specimen_file):

	# Parse cell file
	df_gene = pd.read_csv(plasma_file, sep='\t')

	# Parse subject file for IDs
	df_specimen = pd.read_csv(specimen_file, sep='\t')[['specimen_id', 'subject_id', 'planned_day_relative_to_boost']]

	# Combine specimen and cell files
	merged_df = pd.merge(df_specimen, df_gene, on='specimen_id')


	## X data:
	# Remove all days greater than 1
	df_prev_days = merged_df[merged_df['planned_day_relative_to_boost'] < 3]

	# df_prev_days = df_prev_days.drop_duplicates(subset=['specimen_id', 'planned_day_relative_to_boost','tpm'])

	## Y data:
	# Selects rows with the answer: 14-days, PT, IgG
	planned_day3 = merged_df['planned_day_relative_to_boost'] == 3
	celltype_name   = merged_df['versioned_ensembl_gene_id'] == 'ENSG00000277632.1'
	df_day1 = merged_df[planned_day3 & celltype_name]
	
	df = pd.concat([df_prev_days, df_day1], ignore_index=True)
		
	df['gene_id/day'] = df['versioned_ensembl_gene_id'].astype(str) + '+' + df['planned_day_relative_to_boost'].astype(str) + 'day'
	
	# At this point, subject and specimen are the same so I'm removing specimen_id
	del df['specimen_id']
	del df['planned_day_relative_to_boost']
	del df['versioned_ensembl_gene_id']
	del df['raw_count']
	# df = df.drop_duplicates(subset=['subject_id', 'tpm'])
	return df

# ----------------------------------------------------------

plasma_file2020 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2020LD_pbmc_gene_expression.tsv'
specimen_file2020 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2020LD_specimen.tsv'

plasma_file2021 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2021LD_pbmc_gene_expression.tsv'
specimen_file2021 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/2021LD_specimen.tsv'

plasma_file2022 = './data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/prediction_data/2022BD_pbmc_gene_expression.tsv'
specimen_file2022 = 'data/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/prediction_data/2022BD_specimen.tsv'


df_2020 = parse_files(plasma_file2020, specimen_file2020)
df_2021 = parse_files(plasma_file2021, specimen_file2021)
df_2022 = parse_files(plasma_file2022, specimen_file2022)

# Concatenate 2020 and 2021
df_raw = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)

# Reshaping data 
df_xy = df_raw.pivot(index='subject_id', columns='gene_id/day', values='tpm')

# Removing columns where data is missing
df_xy_temp = df_xy.dropna(axis=1)

subject_ids_2022 = list(set(df_2022['subject_id'].to_list()))

df_reshaped_2022 = df_xy_temp[df_xy_temp.index.isin(subject_ids_2022)]

# Remove subjects that don't have the target gene on day 0
df_xy = df_xy[df_xy['ENSG00000277632.1+0day'].isna() == False]

# Removing rows where Y-data is missing
df_xy = df_xy.dropna(subset=['ENSG00000277632.1+3day'])

# Removing columns where data is missing
df_xy = df_xy.dropna(axis=1)

common_columns = list(set(df_xy.columns).intersection(set(df_reshaped_2022.columns)))

df_xy_tmp = df_xy[common_columns]
df_xy = pd.merge(df_xy_tmp, df_xy['ENSG00000277632.1+3day'], left_index=True, right_index=True)


# Reordering to make target column last column
except_target = [x for x in list(df_xy.columns) if x != 'ENSG00000277632.1+3day']

df = df_xy[except_target + ['ENSG00000277632.1+3day']]

# Fold change
df['ENSG00000277632.1+0day'] = df['ENSG00000277632.1+0day'] + 0.001 # fixes division by 0
df['ENSG00000277632.1+3day'] = df['ENSG00000277632.1+3day']/df['ENSG00000277632.1+0day']


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

df_reshaped_2022['task3.2'] = Y_pred
df_reshaped_2022 = df_reshaped_2022.sort_values('task3.2', ascending=False)
df_reshaped_2022['task3.2'].to_csv('output/task/task3.2_predictions.csv')

