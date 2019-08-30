import numpy as np


def load_users_items_file(path):
    file = open(path, 'r')
    users, items, rates = list(), list(), list()
    for line in file:
        split = line.strip().split(',')
        uid, iid, rate = split[0], split[1], split[2]
        users.append(uid)
        items.append(iid)
        rates.append(rate)
    users.pop(0)
    items.pop(0)
    rates.pop(0)
    users = [int(u) for u in users]
    items = [int(i) for i in items]
    rates = [int(r) for r in rates]
    return (np.array(users), np.array(items), np.array(rates))


if __name__ == '__main__':
    path = '../data/full_data_v3/SO_without_negative_train_test/'
    dataset = 'train_all_new.csv'

    users, items, rates = load_users_items_file(path=path + dataset)
    print(len(users), len(items), len(rates))
