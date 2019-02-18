from sklearn.feature_extraction.text import TfidfVectorizer


def load_tag(path_file, data_type):
    lines = list(open(path_file, 'r').readlines())
    dicts = dict()
    for l in lines:
        split_l = l.strip().split(',')
        id_, tags = data_type + '_' + split_l[0], split_l[1].split(';')
        tags = [t.strip() for t in tags]
        dicts[id_] = tags
    return dicts


def load_user_object(path_file, data_type):
    lines = list(open(path_file, 'r').readlines())
    dicts = dict()
    for l in lines:
        user_id, obj_id = data_type + '_' + l.strip().split(',')[0], data_type + '_' + l.strip().split(',')[1]
        if user_id in dicts:
            value = dicts[user_id]
            value.append(obj_id)
            dicts[user_id] = value
        else:
            dicts[user_id] = [obj_id]
    return dicts


def load_user_tag(user_obj, obj_tag):
    user_tag = dict()
    for user_id in user_obj.keys():
        objs = user_obj[user_id]
        tags = list()
        for obj in objs:
            tags += obj_tag[obj]
        user_tag[user_id] = tags
    return user_tag


def merge_userTag(user_tag, data_type):
    if data_type == 'so_gh':
        so_user_tag, gh_user_tag = user_tag
        so_userID, gh_userID = sorted(so_user_tag.keys()), sorted(gh_user_tag.keys())
        so_userTag, gh_userTag = [so_user_tag[u] for u in so_userID], [gh_user_tag[u] for u in gh_userID]

        all_userID = so_userID + gh_userID
        all_userTag = so_userTag + gh_userTag
        return all_userID, all_userTag
    else:
        print('You need to give the correct data type')
        exit()


def load_user_features(user_tag, ftr_type='tf-idf', data_type='so_gh'):
    if data_type == 'so_gh':
        uID, uTag = merge_userTag(user_tag=user_tag, data_type=data_type)
        uTag = [' '.join(t).strip() for t in uTag]
        if ftr_type == 'tf-idf':
            vectorizer = TfidfVectorizer()
            u_ftr = vectorizer.fit_transform(uTag)
        return uID, u_ftr.todense()


def load_label_data():
    print('hello')



if __name__ == '__main__':
    question_path_file = '../data/SO_GH/question_tag.csv'
    user_question_path_file = '../data/SO_GH/user_question.csv'
    repository_path_file = '../data/SO_GH/repository_tag.csv'
    user_repository_path_file = '../data/SO_GH/user_repository.csv'

    user_question = load_user_object(path_file=user_question_path_file, data_type='so')
    question_tag = load_tag(path_file=question_path_file, data_type='so')
    so_user_tag = load_user_tag(user_obj=user_question, obj_tag=question_tag)

    user_repo = load_user_object(path_file=user_repository_path_file, data_type='gh')
    repo_tag = load_tag(path_file=repository_path_file, data_type='gh')
    gh_user_tag = load_user_tag(user_obj=user_repo, obj_tag=repo_tag)

    user_tag = (so_user_tag, gh_user_tag)
    u_id, u_ftr = load_user_features(user_tag=user_tag)
    print(len(u_id), u_ftr.shape)
    # print(len(question_tag), len(repo_tag), len(user_question))
