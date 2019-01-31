import joblib
import torch as tr

branch_comments_embedded_text_df_train3 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train3\\branch_comments_embedded_text_df_train.pkl')
branch_comments_embedded_text_df_train0 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train0\\branch_comments_embedded_text_df_train.pkl')

branch_comments_features_df_train0 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train0\\branch_comments_features_df_train.pkl')
branch_comments_user_profiles_df_train0 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train0\\branch_comments_user_profiles_df_train.pkl')

branch_comments_features_df_train3 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train3\\branch_comments_features_df_train.pkl')
branch_comments_user_profiles_df_train3 = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\features_to_use\\train3\\branch_comments_user_profiles_df_train.pkl')


# branches_lengths_list_train = joblib.load('C:\\Users\\ssheiba\\Desktop\\MASTER\\NLP_SC_CMV\\train1\\branches_lengths_list_train.txt')

print('check')

for row in branch_comments_embedded_text_df_train.values:
    row_i = 0
    for column in row:
        col = 0
        try:
            tensor = tr.Tensor([column])
        except:
            print(column)
            print("row is: ", row_i)
            print("col is: ", col)
        col +=1
    row_i+=1