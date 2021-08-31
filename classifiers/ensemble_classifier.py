# Creating weighted ensembler using the given models by creating single dataframe
# the weights are assigned by manually evaluating classification reports of every model and picking out a value that
# does not throw off the results of the overall sum

import pandas as pd


# creates the column in the new dataframe using existing df outputs
def create_all_output_df(df, df_add, col_to_add, model_name):
    for index, row in df.iterrows():
        df.at[index, model_name] = df_add.loc[row['file_name'] == df_add['file_name']][col_to_add].item()
    return df


# read all output files
df_context = pd.read_csv('../outputs/context_clssifier_test.csv')
df_effnet = pd.read_csv('../outputs/effnet_meme_multioff_test.csv')
df_bert = pd.read_csv('../outputs/bert_meme_fox_test.csv')
df_ml_dt = pd.read_csv('../outputs/dt_multioff_fox.csv')
df_ml_nb = pd.read_csv('../outputs/nb_multioff_fox.csv')
df_ml_svc = pd.read_csv('../outputs/SVC_multioff_fox.csv')
df_ml_knn = pd.read_csv('../outputs/knn_multioff_fox.csv')

# create new ensemble dataframe
df_ensemble = pd.DataFrame(
    columns=['file_name', 'actual_class_val', 'context_classifier', 'effnet', 'bert', 'dt', 'nb', 'SVC', 'knn',
             'ensembled_results_with_ml', 'ensembled_results_without_ml', 'ensembled_results_bert_effnet'])

# add values to new df using all other output values
df_ensemble['file_name'] = df_context['file_name']
df_ensemble['actual_class_val'] = df_context[' actual_class_val']

df_ensemble = create_all_output_df(df_ensemble, df_context, 'pred_class_val', 'context_classifier')
df_ensemble = create_all_output_df(df_ensemble, df_effnet, 'pred_class_val', 'effnet')
df_ensemble = create_all_output_df(df_ensemble, df_bert, 'pred_class_val', 'bert')
df_ensemble = create_all_output_df(df_ensemble, df_ml_dt, 'pred_class_val', 'dt')
df_ensemble = create_all_output_df(df_ensemble, df_ml_nb, 'pred_class_val', 'nb')
df_ensemble = create_all_output_df(df_ensemble, df_ml_svc, 'pred_class_val', 'SVC')
df_ensemble = create_all_output_df(df_ensemble, df_ml_knn, 'pred_class_val', 'knn')

# in new df column, add ensembled results for only three given models with specific weights
for index, row in df_ensemble.iterrows():
    val = ((0.61 * row['context_classifier']) + (0.85 * row['effnet']) + (1 * row['bert'])) / .82
    if val > 0.5:
        val = 1
    else:
        val = 0
    df_ensemble.at[index, 'ensembled_results_without_ml'] = val

# in new df column, add ensembled results for all given models with specific weights
for index, row in df_ensemble.iterrows():
    val = ((0.61 * row['context_classifier']) + (0.85 * row['effnet']) + (1 * row['bert']) + (0.2 * row['dt']) + (
            0.09 * row['nb']) + (0.2 * row['SVC']) + (0.2 * row['knn'])) / 0.45
    if val > 0.5:
        val = 1
    else:
        val = 0
    df_ensemble.at[index, 'ensembled_results_with_ml'] = val

# in new df column, add ensembled results for bert and effnet classifiers
for index, row in df_ensemble.iterrows():
    val = ((0.85 * row['effnet']) + (1 * row['bert'])) / 0.925
    if val > 0.5:
        val = 1
    else:
        val = 0
    df_ensemble.at[index, 'ensembled_results_bert_effnet'] = val

# save new df to outputs
df_ensemble.to_csv('../outputs/ensembled_results.csv')
