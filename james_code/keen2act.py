from utilities_ver2 import loadInteractions, loadUserFeatures, loadItemFeatures, loadInteractions_keenfunc_thres, \
    mini_batch, load_data
from lightfm import LightFM
import pickle
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import torch.nn as nn
import torch


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        output = self.fc2(self.fc1(x))
        return output


def save_ranking_model(path, model):
    print('Saving ranking model...')
    with open(path, 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)


def load_ranking_model(path):
    print('Loading ranking model...')
    return pickle.load(open(path, 'rb'))


def train_ranking_model(interactions, user_features, item_features):
    model = LightFM(learning_rate=0.05, loss='bpr')
    print('Training ranking model...')
    model.fit(interactions=interactions, user_features=user_features, item_features=item_features, epochs=50,
              verbose=True, num_threads=25)
    print('Complete...')
    return model


def predict_ranking_model(users, items, user_features, item_features, model):
    print('Prediction ranking model...')
    return model.predict(user_ids=users, item_ids=items, user_features=user_features, item_features=item_features,
                         num_threads=10)


def initial_threshold(items):
    threshold_values = dict()
    for item in items.keys():
        if item not in threshold_values.keys():
            threshold_values[item] = sum(items[item]) / float(len(items[item]))  # theta values
    return threshold_values


def construct_data(items, thresholds):
    print('Construcing data for threshold model...')
    data = list()
    for item in items.keys():
        ranking_scores = items[item]
        for r in ranking_scores:
            data.append(np.array([r, thresholds[item]]))
    return np.array(data)


def save_threshold_model(path, model):
    torch.save(model.state_dict(), path)


def load_threshold_model(path):
    model = torch.load(path)
    return model


def train_threshold_model(ranking_model, path_interaction_threshold, path_user_features, path_item_features):
    users, items, labels = load_data(path_interaction_threshold)
    u_features = load_data(path_user_features)
    i_features = load_data(path_item_features)
    # scores = predict_ranking_model(users=users, items=items, user_features=u_features, item_features=i_features,
    #                                model=ranking_model)
    # # print(len(users), len(items), len(labels))
    # # print(u_features.shape, i_features.shape)
    # # print(len(scores))
    # # # Y = labels
    # # # Y = Y.tolist()
    # # # Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    # # # Y_neg = [i for i in range(len(Y)) if Y[i] == 0]
    # # # print(len(Y_pos), len(Y_neg))
    # # exit()
    # # users, items, labels = loadInteractions_keenfunc_thres(path_interaction)
    # # u_features = loadUserFeatures(path_user_features, n_users)
    # # i_features = loadItemFeatures(path_item_features, n_items, n_tags)
    scores = predict_ranking_model(users=users, items=items, user_features=u_features, item_features=i_features,
                                   model=ranking_model)

    scores = list(scores)
    item_dicts = dict()
    for i, r in zip(items, scores):
        if i not in item_dicts.keys():
            item_dicts[i] = [r]
        else:
            item_dicts[i].append(r)

    initial_values = initial_threshold(items=item_dicts)
    X = construct_data(items=item_dicts, thresholds=initial_values)  # first column is ranking, second is threshold
    if 'without' in path_interaction:  # non-negative values
        Y = np.array([1] * X.shape[0])
    else:
        Y = labels
    batches = mini_batch(X=X, Y=Y)
    print(len(scores))
    print(len(batches))
    exit()
    model = NeuralNet(input_dim=1, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(batches)):
            batch = batches[i]
            batch_X, batch_y = batch
            batch_X, batch_y = torch.tensor(batch_X).float(), torch.tensor(batch_y).float()
            batch_ranking = batch_X[:, 1]
            batch_threshold = torch.reshape(batch_X[:, 1], (batch_X.shape[0], 1))
            outputs = torch.reshape(model.forward(x=batch_threshold), (batch_X.shape[0],))
            outputs = sigmoid(batch_ranking - outputs)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, total_loss / len(batches)))

    return model


if __name__ == '__main__':
    ##############################################################################################
    # Step 1a: ranking model
    ##############################################################################################

    # # Loading iteraction, users graph, and items graph
    # ##############################################################################################
    # # SO
    # # path_interaction = '../data/full_data_v3/SO_without_negative_train_test/train_all_new.csv'
    # # path_user_features = '../data/full_data_v3/so_user_user_graph.csv'
    # # path_item_features = '../data/full_data_v3/so_item_graph.csv'
    #
    # path_interaction = '../data/full_data_v5/so/train.csv'
    # path_user_features = '../data/full_data_v5/so/so_user_user_graph.csv'
    # path_item_features = '../data/full_data_v5/so/so_item_graph.csv'
    # n_users, n_items, n_tags = 23612, 1020809, 30354
    #
    # interactions = loadInteractions(path_interaction, n_users, n_items)
    # u_features = loadUserFeatures(path_user_features, n_users)
    # i_features = loadItemFeatures(path_item_features, n_items, n_tags)
    # print('Shape of interactions, user features, and item features...')
    # print(interactions.shape, u_features.shape, i_features.shape)
    # ##############################################################################################
    #
    # # Training model
    # r_model = train_ranking_model(interactions=interactions, user_features=u_features, item_features=i_features)
    #
    # # Saving model
    # ##############################################################################################
    # path_r_model = 'keenfunc_ranking.pickle'
    # save_ranking_model(path=path_r_model, model=r_model)
    # # r_model = load_ranking_model(path=path_r_model)
    # # ##############################################################################################
    #
    # # Predict
    # ##############################################################################################
    # path_test = '../data/full_data_v3/SO_without_negative_train_test/test_answer_new.csv'
    # test_users, test_items, y_true = loadTest(path_test)
    # y_pred = predict_ranking_model(users=test_users, items=test_items, user_features=u_features,
    #                                item_features=i_features, model=r_model)
    # y_pred = np.array([y_pred])
    # y_true = np.array([y_true])
    # score = label_ranking_average_precision_score(y_true=y_true, y_score=y_pred)
    # print(score)
    # exit()

    ##############################################################################################
    # Step 1b: threshold model
    ##############################################################################################
    # SO

    # path_interaction = '../data/full_data_v3/SO_without_negative_train_test/train_all_new.csv'
    # path_user_features = '../data/full_data_v3/so_user_user_graph.csv'
    # path_item_features = '../data/full_data_v3/so_item_graph.csv'

    # path_interaction = '../data/full_data_v5/so/train.csv'
    # path_user_features = '../data/full_data_v5/so/so_user_user_graph.csv'
    # path_item_features = '../data/full_data_v5/so/so_item_graph.csv'

    path_interaction = './data/so_interactions.pickle'
    path_interaction_threshold = './data/so_interactions_threshold.pickle'
    path_user_features = './data/so_user_features.pickle'
    path_item_features = './data/so_item_features.pickle'
    n_users, n_items, n_tags = 23612, 1020809, 30354

    # # Loading ranking model
    path_r_model = 'keenfunc_ranking.pickle'
    r_model = load_ranking_model(path=path_r_model)

    # # Threshold model
    threshold_model = train_threshold_model(ranking_model=r_model,
                                            path_interaction_threshold=path_interaction_threshold,
                                            path_user_features=path_user_features,
                                            path_item_features=path_item_features)
