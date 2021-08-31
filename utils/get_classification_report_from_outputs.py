# prints and saves the classification reports using outputs of classification reports
import pandas as pd
from sklearn import metrics


# only prints classification reports
def print_classification_report(model_output_path, pred_col='pred_class_val', actual_col=' actual_class_val'):
    print('Classification report: ')

    # read test csv and create df to store results
    df = pd.read_csv(model_output_path)
    print(df.columns)
    print(metrics.classification_report(df[pred_col], df[actual_col]))

    return pd.DataFrame(metrics.classification_report(df[pred_col], df[actual_col], output_dict=True)).transpose()


# prints and saves classification reports using the given filepaths
def save_classification_report(model_output_path, file_save_path, pred_col='pred_class_val',
                               actual_col=' actual_class_val'):
    df = print_classification_report(model_output_path, pred_col, actual_col)
    df.to_csv(file_save_path)

# save_classification_report('../outputs/effnet_meme_multioff_test.csv', '../classification_reports/effnet_meme_multioff_images_test.csv')
# save_classification_report('../outputs/bert_meme_fox_test.csv', '../classification_reports/bert_meme_fox_text_test.csv', actual_col='actual_value')
# save_classification_report('../outputs/knn_multioff_fox.csv', '../classification_reports/knn_multioff_fox_text_test.csv')
# save_classification_report('../outputs/dt_multioff_fox.csv', '../classification_reports/dt_multioff_fox_text_test.csv')
# save_classification_report('../outputs/nb_multioff_fox.csv', '../classification_reports/nb_multioff_fox_text_test.csv')
# save_classification_report('../outputs/SVC_multioff_fox.csv', '../classification_reports/SVC_multioff_fox_text_test.csv')
# save_classification_report('../outputs/context_clssifier_test.csv', '../classification_reports/context_classifier_text_test.csv')
# save_classification_report('../outputs/ensembled_results.csv', '../classification_reports/ensembled_with_ml.csv', pred_col='ensembled_results_with_ml', actual_col='actual_class_val')
# save_classification_report('../outputs/ensembled_results.csv', '../classification_reports/ensembled_without_ml.csv', pred_col='ensembled_results_without_ml', actual_col='actual_class_val')
# save_classification_report('../outputs/ensembled_results.csv', '../classification_reports/ensembled_without_bert_effnet.csv', pred_col='ensembled_results_bert_effnet', actual_col='actual_class_val')
