import zipfile
from urllib.request import urlretrieve
import torch
import pandas as pd
import os


class RecommendationsEnvironment:
    currentStep = 0
    rewards = {
        0: 3,
        1: 0,
        2: -1,
        3: -2,
        4: -3
    }
    data = []
    ratings = []

    def __init__(self, device, directory='data'):
        self.device = device
        self.directory = directory

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        self.downloadMovieLensData()
        self.processData()

    def downloadMovieLensData(self):
        zipPath = os.path.join(self.directory, 'movielens.zip')
        urlretrieve('http://files.grouplens.org/datasets/movielens/ml-100k.zip', zipPath)
        zip_ref = zipfile.ZipFile(zipPath, 'r')
        zip_ref.extractall(self.directory)

    def processData(self):
        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        raw_ratings = pd.read_csv(os.path.join(self.directory, 'ml-100k', 'u.data'), sep='\t', names=ratings_cols, encoding='latin-1')

        user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip']
        raw_users = pd.read_csv(os.path.join(self.directory, 'ml-100k', 'u.user'), sep='|', names=user_cols, encoding='latin-1')
        occupations = raw_users.occupation.unique().tolist()

        movies_cols = ['movie_id', 'title', 'wtf', 'release_date', 'url']
        for i in range(0, 19):
            movies_cols.append('genre_%d' % i)
        raw_movies = pd.read_csv(os.path.join(self.directory, 'ml-100k', 'u.item'), sep='|', names=movies_cols, encoding='latin-1')

        users = {}
        for user in raw_users.values:
            users[user[0]] = self.oneHotEncodeAge(user[1]) + self.oneHotEncodeSex(user[2]) + self.oneHotEncodeOccupation(occupations, user[3])

        genres = {}
        for movie in raw_movies.values:
            genres[movie[0]] = movie[6:].tolist()

        for item in raw_ratings.values:
            self.data.append(users[item[0]] + genres[item[1]])
            self.ratings.append(item[2])

    def oneHotEncodeAge(self, age):
        if age < 10:
            cat = 0
        elif age < 20:
            cat = 1
        elif age < 30:
            cat = 2
        elif age < 40:
            cat = 3
        elif age < 50:
            cat = 4
        elif age < 60:
            cat = 5
        elif age < 70:
            cat = 6
        else:
            cat = 7

        ret = [0] * 8
        ret[cat] = 1
        return ret

    def oneHotEncodeSex(self, sex):
        if sex == 'M':
            return [1, 0]
        return [0, 1]

    def oneHotEncodeOccupation(self, occupations, occupation):
        ret = [0] * len(occupations)
        ret[occupations.index(occupation)] = 1
        return ret

    def _getState(self):
        return torch.tensor([self.data[self.currentStep]], dtype=torch.float32, device=self.device)

    def getInputSize(self):
        return len(self.data[0])

    def reset(self):
        self.currentStep = 0
        return self._getState()

    def step(self, action):
        action += 1
        rating = self.ratings[self.currentStep]
        error = abs(rating - action)
        reward = self.rewards[error]
        self.currentStep += 1
        done = self.currentStep == len(self.data) - 1

        return (self._getState(), reward, done, None)
