import os
import pandas as pd


def return_df():
    data_folder = 'data/dga_data'
    dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, dga_type))]
    print(dga_types)
    my_df = pd.DataFrame(columns=['domain', 'type', 'label'])
    # for dga_type in dga_types:
    #     files = os.listdir(os.path.join(data_folder, dga_type))
    #     for file in files:
    #         with open(os.path.join(data_folder, dga_type, file), 'r') as fp:
    #             domains_with_type = [[(line.strip()), dga_type, 1] for line in fp.readlines()]
    #             appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
    #             my_df = pd.concat([my_df, appending_df], ignore_index=True)

    # with open(os.path.join(data_folder, 'benign.txt'), 'r') as fp:
    #     domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:60000]]
    #     appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
    #     my_df = pd.concat([my_df, appending_df], ignore_index=True)

    with open(os.path.join(data_folder, 'iid_data_1.txt'), 'r') as fp:
        domains_with_type = [line.strip().split('\t') for line in fp.readlines()]
        print(domains_with_type)
        appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
        my_df = pd.concat([my_df, appending_df], ignore_index=True)
    
    
    print (my_df)

return_df()
