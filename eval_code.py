import pandas as pd
import sys

### Instructions for participants : 
'''
During training participants can split training file into two files (1st : submission file ; 2nd : dependent variable file) the details of which are mentioned below.

You can use this code to measure train dataset performance of the model.
### Datasets required:
This script takes in 2 files as follows:

submission.csv -> This contains the customer, merchant & predicted_score columns
dep_var.csv    -> This contains customer,merchant,ind_recommended and activation
NOTE: You do not need to necessarily name your files as submission.csv, dep_var.csv

Please ensure that the predicted_score column does not have any null columns and the column names are exactly matching as above.
Please ensure that all these files are stored as ',' separated csv files.

### How to use:
To use this, first open the command line terminal, and call evaluation code script by passing the locations of submission and actual files respectively.
Sample example of using commandline for running the script: 

python path_to_Evaluation_Code path_to_submission_file path_to_DepVar_file

You can also use this function separately by manually overriding the code for loading the submission.csv and dep_var.csv

'''


#creating custom function for MSB to just return Top 10 rank values of activation
def incr_act_top10(input_df,pred_col,cm_key='customer',treated_col='ind_recommended',actual_col='activation'):
    
	#for correcting variable types
    input_df[[treated_col, actual_col, pred_col]] = input_df[[treated_col, actual_col, pred_col]].apply(pd.to_numeric, errors='coerce')
	
    input_df['rank_per_cm1'] = input_df.groupby(cm_key)[pred_col].rank(method='first', ascending=False)
    
    input_df = input_df.loc[input_df.rank_per_cm1 <= 10,:]
    
    agg_df = input_df.groupby(treated_col,as_index=False).agg({actual_col:'mean'})
    agg_df.columns = [treated_col,'avg_30d_act']
    
    print(agg_df)
    recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==1,'avg_30d_act'])
    not_recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==0,'avg_30d_act'])
    
    return (recommended_avg_30d_act-not_recommended_avg_30d_act)
	
# sys.argv takes in the arguments that you pass from terminal. First argument is the location of Evaluation code file, Second argument is the location of submission file and Third argument is the location of Dependent Variable file
if len(sys.argv) != 3:
  sys.exit("Please pass two files only as mentioned in the Instructions.")

# Location of submission file. Header here should include customer, merchant & predicted_score. The file should be comma separated.
input_address = sys.argv[1] 
try:
  df_input = pd.read_csv(input_address,sep=",",header=0, dtype = {'predicted_score': float})
except:
  sys.exit("Please ensure that predicted_score column has all non-null numeric values")


# round off scores to 10 decimal points
df_input['predicted_score'] = df_input['predicted_score'].round(10)


# groupby customer, merchant and max score
df_input = df_input.groupby(['customer', 'merchant'], as_index = False)['predicted_score'].agg('max')

### Location of Dependent Variable file. Header here should include customer, merchant, ind_recommended & activation. The file should be comma separated.
round_eval = sys.argv[2]
df_round = pd.read_csv(round_eval,sep=",",header=0).drop_duplicates()


# merging predicted file and dependent variable file
eval_data = pd.merge(df_round,df_input,on=['customer','merchant'],how='inner').drop_duplicates()
# deleting the rows having null value in predicted_score
eval_data = eval_data[~(eval_data['predicted_score'].isna())]


if df_round.shape[0] != eval_data.shape[0]:
  sys.exit("The number of Unique Customer x Merchant in Submission data do not match with number of Unique Customer x Merchant in Dependent Variable data")
else:
  print('Input Files are Correct')

final_score = round(incr_act_top10(input_df=eval_data,pred_col='predicted_score',cm_key='customer',treated_col='ind_recommended',actual_col='activation'), 7)

print('Incremental Activation Rate for Top 10 ranked Merchants(dataset level): ', final_score)