

def create_dataset_dict(dataset: str, all_data_dict: dict) -> dict:
    """
    The function get dataset name and dict and return a dict for this dataset
    :param dataset: the name of the data set (train, testi, valid)
    :param all_data_dict: the dict of all the data
    :return:
    """

    return {
        'branch_comments_embedded_text_df': all_data_dict[dataset][f'branch_comments_embedded_text_df'],
        'branch_comments_features_df': all_data_dict[dataset][f'branch_comments_features_df'],
        'branch_comments_user_profiles_df': all_data_dict[dataset][f'branch_comments_user_profiles_df'],
        'branch_submission_dict': all_data_dict[dataset][f'branch_submission_dict'],
        'submission_data_dict': all_data_dict[dataset][f'submission_data_dict'],
        'branch_deltas_data_dict': all_data_dict[dataset][f'branch_deltas_data_dict'],
        'branches_lengths_list': all_data_dict[dataset][f'branches_lengths_list'],
        'len_df': all_data_dict[dataset]['len_df']
    }


def get_model_layer_sizes(data_dict: dict) -> dict:
    """
    The function get dict with the data and return the sizes for the model's layers
    :param data_dict:
    :return:
    """

    return {
        'lstm_text': data_dict['branch_comments_embedded_text_df'].iloc[0, 0],
        'lstm_comments': data_dict['branch_comments_features_df'].iloc[0, 0],
        'lstm_users': data_dict['branch_comments_user_profiles_df'].iloc[0, 0],
        'input_size_text_sub': data_dict['branch_comments_embedded_text_df'].iloc[0, 0],
        'input_size_sub_features':
            len(data_dict['submission_data_dict'][list(data_dict['submission_data_dict'].keys())[0]][1]) +
            len(data_dict['branch_submission_dict'][list(data_dict['branch_submission_dict'].keys())[0]][1]),
        'input_size_sub_profile_features':
            len(data_dict['submission_data_dict'][list(data_dict['submission_data_dict'].keys())[0]][2])

    }
