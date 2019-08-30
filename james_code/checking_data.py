import csv


def load_items(i_train):
    print('Loading Interaction without activities...')
    # Sparse Matrix of size U x (U + (IA))
    items = list()
    with open(i_train, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            des = int(row[1])
            items.append(des)
            # make sure it is symmetry
    return items


if __name__ == '__main__':
    i_train = '../data/full_data_v7/so/train.csv'
    train_items = load_items(i_train=i_train)

    i_test = '../data/full_data_v7/so/test.csv'
    test_items = load_items(i_train=i_test)

    print('Training items: ', len(train_items), '----', 'Testing items: ', len(test_items))
    for t in test_items:
        if t not in train_items:
            print('Items', t, 'not in training data')
            break