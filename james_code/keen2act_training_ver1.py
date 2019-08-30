from load_data_ver2 import load_data, save_data
from lightfm import LightFM
import pickle
import numpy as np
from utilities_ver2 import mini_batch
import torch.nn as nn
import torch


def save_ranking_model(path, model):
    print('Saving ranking model...')
    with open(path, 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)


def train_keen_interactions_model(interactions, user_features, item_features, n_items):
    # SO
    # L_RATE = 0.05
    # L_LOSS = 'warp'
    # ALPHA = 1E-09
    # NUM_COMPONENTS = 180
    # NUM_THREADS = 20
    # EPOCH = 50
    # MAX_SAMPLE = int(n_items * 0.01)

    # GH
    L_RATE = 0.05
    L_LOSS = 'warp'
    ALPHA = 1E-07
    NUM_COMPONENTS = 50
    NUM_THREADS = 20
    EPOCH = 10
    MAX_SAMPLE = int(n_items * 0.001)

    model = LightFM(no_components=NUM_COMPONENTS,
                    learning_rate=L_RATE,
                    loss=L_LOSS,
                    item_alpha=ALPHA,
                    user_alpha=ALPHA,
                    max_sampled=MAX_SAMPLE)
    print('---------------------------------------')
    print('L_RATE: User_ALL, Item_ALL')
    print('L_RATE:' + str(L_RATE))
    print('L_LOSS:' + str(L_LOSS))
    print('ALPHA:' + str(ALPHA))
    print('NUM_COMPONENTS:' + str(NUM_COMPONENTS))
    print('EPOCH:' + str(EPOCH))
    print('MAX_SAMPLE:' + str(MAX_SAMPLE))
    print('---------------------------------------')

    # Fitting model
    model.fit(interactions, user_features=user_features, item_features=item_features, epochs=EPOCH, verbose=True,
              num_threads=NUM_THREADS)
    return model


def initial_threshold(items):
    threshold_values = dict()
    for item in sorted(items.keys()):
        if item not in threshold_values.keys():
            item_values = items[item]
            item_length = len(items[item])
            threshold_values[item] = sum(item_values) / float(item_length)  # theta values
    return threshold_values


def construct_data(items, thresholds):
    print('Construcing data for threshold model...')
    data = list()
    for item in sorted(items.keys()):
        ranking_scores = items[item]
        for r in ranking_scores:
            data.append(np.array([r, thresholds[item]]))
    return np.array(data)


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        output = self.fc2(self.fc1(x))
        return output


def train_keen_threshold_model(keen_interactions_model, interactions, user_features, item_features, num_epochs):
    users, items, labels = interactions
    users_items_pred_rank = keen_interactions_model.predict(user_ids=np.array(users), item_ids=np.array(items),
                                                            user_features=user_features,
                                                            item_features=item_features,
                                                            num_threads=20)

    users_items_pred_rank = list(users_items_pred_rank)
    items_users_ranking = dict()
    for i, r in zip(items, users_items_pred_rank):
        if i not in items_users_ranking.keys():
            items_users_ranking[i] = [r]
        else:
            items_users_ranking[i].append(r)

    initial_items_threshold = initial_threshold(items=items_users_ranking)
    X = construct_data(items=items_users_ranking,
                       thresholds=initial_items_threshold)  # first column is ranking, second is threshold
    Y = np.array(labels)
    batches = mini_batch(X=X, Y=Y)
    model = NeuralNet(input_dim=1, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(batches)):
            batch = batches[i]
            batch_X, batch_y = batch
            batch_X, batch_y = torch.tensor(batch_X).float(), torch.tensor(batch_y).float()
            batch_ranking = batch_X[:, 0]  # get ranking
            batch_threshold = torch.reshape(batch_X[:, 1], (batch_X.shape[0], 1))  # get threshold
            outputs = torch.reshape(model.forward(x=batch_threshold), (batch_X.shape[0],))
            outputs = sigmoid(batch_ranking - outputs)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, total_loss / len(batches)))
    items_threshold = np.array([initial_items_threshold[item] for item in sorted(initial_items_threshold.keys())])
    items_threshold = torch.tensor(items_threshold).float()
    items_threshold = torch.reshape(items_threshold.float(), (items_threshold.shape[0], 1))
    items_threshold = model.forward(items_threshold)
    items_threshold = list(items_threshold.cpu().detach().numpy())
    new_items_threshold, cnt = dict(), 0
    for item in sorted(initial_items_threshold.keys()):
        new_items_threshold[item] = items_threshold[cnt][0]
        cnt += 1
    return model, new_items_threshold


if __name__ == '__main__':
    # Stack Overflow (SO)
    n_users, n_items = 23612, 1020809
    ##############################################################################################
    # KEEN MODEL
    # ##############################################################################################
    # # Step 1a: train interaction model
    # ##############################################################################################
    # # Loading data
    # path_interaction_non_activities = './data_ver1/so/interactions_non_activities.pickle'
    # interactions_non_activities = load_data(path=path_interaction_non_activities)[0]  # it's a sparse matrix
    # print('Interaction without activities', interactions_non_activities.shape)
    #
    # path_i_features = './data_ver1/so/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users and Items for Keen step
    # keen_interactions_model = train_keen_interactions_model(interactions=interactions_non_activities,
    #                                                         user_features=u_features_CO,
    #                                                         item_features=i_features,
    #                                                         n_items=n_items)
    # path_keen_interactions_model = './model/so/keen_interactions_model.pickle'
    # save_data(path=path_keen_interactions_model, data=keen_interactions_model)

    ##############################################################################################
    # Step 1b: train threshold model
    ##############################################################################################
    # path_interaction_non_activities = './data_ver1/so/interactions_non_activities.pickle'
    # interactions_non_activities = load_data(path=path_interaction_non_activities)[1]  # users, items, labels
    # print('Interaction without activities', len(interactions_non_activities))
    #
    # path_i_features = './data_ver1/so/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # path_keen_interactions_model = './model/so/keen_interactions_model.pickle'
    # keen_interactions_model = load_data(path=path_keen_interactions_model)
    # num_epochs = 15
    # keen_threshold_model, keen_items_threshold = train_keen_threshold_model(
    #     keen_interactions_model=keen_interactions_model,
    #     interactions=interactions_non_activities,
    #     user_features=u_features_CO,
    #     item_features=i_features,
    #     num_epochs=num_epochs)
    # torch.save(keen_threshold_model.state_dict(), './model/so/keen_threshold_model_epoch_' + str(num_epochs) + '.torch')
    # save_data(path='./model/so/keen_items_threshold_epoch_' + str(num_epochs) + '.pickle', data=keen_items_threshold)
    # exit()

    ##############################################################################################
    ##############################################################################################
    # ACT MODEL
    # ##############################################################################################
    # # Stack Overflow (SO)
    # n_users, n_items = 23612, 1020809
    # # First activities
    # # Loading data
    # path_interactions_with_first_activities_for_ACT = './data_ver1/so/interactions_with_first_activities_for_ACT.pickle'
    # # it's a sparse matrix
    # path_interactions_with_first_activities_for_ACT = load_data(path=path_interactions_with_first_activities_for_ACT)[0]
    # print('Interaction without activities', path_interactions_with_first_activities_for_ACT.shape)
    #
    # path_i_features = './data_ver1/so/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users, Items, Activities for Act step
    # act_interactions_model_first_activities = train_keen_interactions_model(
    #     interactions=path_interactions_with_first_activities_for_ACT,
    #     user_features=u_features_CO,
    #     item_features=i_features,
    #     n_items=n_items)
    # path_act_interactions_model_first_activities = './model/so/act_interactions_model_first_activities.pickle'
    # save_data(path=path_act_interactions_model_first_activities, data=act_interactions_model_first_activities)

    # Second activities
    # Loading data
    # n_users, n_items = 23612, 1020809
    # path_interactions_with_second_activities_for_ACT = './data_ver1/so/interactions_with_second_activities_for_ACT.pickle'
    # # it's a sparse matrix
    # path_interactions_with_second_activities_for_ACT = load_data(path=path_interactions_with_second_activities_for_ACT)[
    #     0]
    # print('Interaction without activities', path_interactions_with_second_activities_for_ACT.shape)
    #
    # path_i_features = './data_ver1/so/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users, Items, Activities for Act step
    # act_interactions_model_second_activities = train_keen_interactions_model(
    #     interactions=path_interactions_with_second_activities_for_ACT,
    #     user_features=u_features_CO,
    #     item_features=i_features,
    #     n_items=n_items)
    # path_act_interactions_model_second_activities = './model/so/act_interactions_model_second_activities.pickle'
    # save_data(path=path_act_interactions_model_second_activities, data=act_interactions_model_second_activities)

    ##############################################################################################
    # Github (GH)
    n_users, n_items = 33453, 461931
    ##############################################################################################
    # KEEN MODEL
    # ##############################################################################################
    # # Step 1a: train interaction model
    # ##############################################################################################
    # Loading data
    # path_interaction_non_activities = './data_ver1/gh/interactions_non_activities.pickle'
    # interactions_non_activities = load_data(path=path_interaction_non_activities)[0]  # it's a sparse matrix
    # print('Interaction without activities', interactions_non_activities.shape)
    #
    # path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users and Items for Keen step
    # keen_interactions_model = train_keen_interactions_model(interactions=interactions_non_activities,
    #                                                         user_features=u_features_CO,
    #                                                         item_features=i_features,
    #                                                         n_items=n_items)
    # path_keen_interactions_model = './model/gh/keen_interactions_model.pickle'
    # save_data(path=path_keen_interactions_model, data=keen_interactions_model)

    ##############################################################################################
    # Step 1b: train threshold model
    ##############################################################################################
    path_interaction_non_activities = './data_ver1/gh/interactions_non_activities.pickle'
    interactions_non_activities = load_data(path=path_interaction_non_activities)[1]  # users, items, labels
    print('Interaction without activities', len(interactions_non_activities))

    path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    i_features = load_data(path=path_i_features)
    print('Item features', i_features.shape)

    path_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    u_features_CO = load_data(path=path_user_features_CO)
    print('User features', u_features_CO.shape)

    path_keen_interactions_model = './model/gh/keen_interactions_model.pickle'
    keen_interactions_model = load_data(path=path_keen_interactions_model)
    num_epochs = 15
    keen_threshold_model, keen_items_threshold = train_keen_threshold_model(
        keen_interactions_model=keen_interactions_model,
        interactions=interactions_non_activities,
        user_features=u_features_CO,
        item_features=i_features,
        num_epochs=num_epochs)
    torch.save(keen_threshold_model.state_dict(), './model/gh/keen_threshold_model_epoch_' + str(num_epochs) + '.torch')
    save_data(path='./model/gh/keen_items_threshold_epoch_' + str(num_epochs) + '.pickle', data=keen_items_threshold)
    exit()

    ##############################################################################################
    ##############################################################################################
    # ACT MODEL
    # ##############################################################################################
    # Github(GH)
    # n_users, n_items = 33453, 461931
    # # # First activities
    # # # Loading data
    # path_interactions_with_first_activities_for_ACT = './data_ver1/gh/interactions_with_first_activities_for_ACT.pickle'
    # # it's a sparse matrix
    # path_interactions_with_first_activities_for_ACT = load_data(path=path_interactions_with_first_activities_for_ACT)[0]
    # print('Interaction without activities', path_interactions_with_first_activities_for_ACT.shape)
    #
    # path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users, Items, Activities for Act step
    # act_interactions_model_first_activities = train_keen_interactions_model(
    #     interactions=path_interactions_with_first_activities_for_ACT,
    #     user_features=u_features_CO,
    #     item_features=i_features,
    #     n_items=n_items)
    # path_act_interactions_model_first_activities = './model/gh/act_interactions_model_first_activities.pickle'
    # save_data(path=path_act_interactions_model_first_activities, data=act_interactions_model_first_activities)

    # Second activities
    # Loading data
    # Github(GH)
    # n_users, n_items = 33453, 461931
    # path_interactions_with_second_activities_for_ACT = './data_ver1/gh/interactions_with_second_activities_for_ACT.pickle'
    # # it's a sparse matrix
    # path_interactions_with_second_activities_for_ACT = load_data(path=path_interactions_with_second_activities_for_ACT)[
    #     0]
    # print('Interaction without activities', path_interactions_with_second_activities_for_ACT.shape)
    #
    # path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)
    # print('Item features', i_features.shape)
    #
    # path_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    # u_features_CO = load_data(path=path_user_features_CO)
    # print('User features', u_features_CO.shape)
    #
    # # Training interaction model between Users, Items, Activities for Act step
    # act_interactions_model_second_activities = train_keen_interactions_model(
    #     interactions=path_interactions_with_second_activities_for_ACT,
    #     user_features=u_features_CO,
    #     item_features=i_features,
    #     n_items=n_items)
    # path_act_interactions_model_second_activities = './model/gh/act_interactions_model_second_activities.pickle'
    # save_data(path=path_act_interactions_model_second_activities, data=act_interactions_model_second_activities)
