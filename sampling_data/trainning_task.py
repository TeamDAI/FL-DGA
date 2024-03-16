from sampling_dga import *
from lstm import *

def start_training_task():
    data_folder = 'data'
    dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]

    print(dga_types)

    my_df = pd.DataFrame(columns=['domain', 'type', 'label'])
    
    for dga_type in dga_types:
        files = os.listdir(f"{data_folder}/{dga_type}")
        for file in files:
            with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                domains_with_type = [[(line.strip()), dga_type, 1] for line in fp.readlines()]
                appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                my_df = pd.concat([my_df, appending_df], ignore_index=True)

    with open(f'{data_folder}/benign.txt', 'r') as fp:
        domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:60000]]
        appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
        my_df = pd.concat([my_df, appending_df], ignore_index=True)

    train_test_df, val_df = train_test_split(my_df, test_size=0.1, shuffle=True) 
    print(train_test_df)
    # Pre-processing
    domains = train_test_df['domain'].to_numpy()
    labels = train_test_df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]

    print(f"Number of samples: {len(encoded_domains)}")
    print(f"One-hot dims: {len(char2ix) + 1}")
    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)

    X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)

    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)
    lr = 3e-4
    epochs = 10

    model = LSTMModel(max_features, embed_size, hidden_size, n_layers).to(device)
    # model = BiLSTM(max_features, embed_size, hidden_size, n_layers).to(device)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

    logging.info("Using device: %s", device)


start_training_task()