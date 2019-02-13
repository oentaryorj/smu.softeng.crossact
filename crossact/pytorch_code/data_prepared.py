def load_tag(path_file):
    lines = list(open(path_file, 'r').readlines())
    dicts = dict()
    for l in lines:
        split_l = l.strip().split(',')
        id_, tags = split_l[0], split_l[1].split(';')
        dicts[id_] = tags
    return dicts


def load_user_object(path_file):
    lines = list(open(path_file, 'r').readlines())
    dicts = dict()
    for l in lines:
        split_l = l.strip().split(',')
        user_id, obj_id = l.strip().split(',')[0], l.strip().split(',')[1]
        if user_id in dicts:
            value = dicts[user_id]
            value.append(obj_id)
            dicts[user_id] = value
        else:
            dicts[user_id] = [obj_id]
        # print(user_id, obj_id)
    exit()



if __name__ == '__main__':
    question_path_file = '../data/SO_GH/question_tag.csv'
    user_question_path_file = '../data/SO_GH/user_question.csv'
    repository_path_file = '../data/SO_GH/repository_tag.csv'
    question_tag = load_tag(path_file=question_path_file)
    repo_tag = load_tag(path_file=repository_path_file)
    user_question = load_user_object(path_file=user_question_path_file)
    print(len(question_tag), len(repo_tag))
