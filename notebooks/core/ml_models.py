
import os
import pickle
import time
from multiprocessing import Process, Manager
from sklearn.tree import DecisionTreeRegressor

class MLModels():
    def __init__(self, db, datasets, models_path):
        """Handles all operations on ML models.

        Args:
            db (ConfigDB): ConfigDB instance.
            datasets (Datasets): Datasets instance.
            models_path (str): local path where the models are stored.
        """
        self.db = db
        self.models_path = models_path
        self.datasets = datasets

        # tracking state about (algorithm, hw, target) that are currently being trained
        self.ongoing_training = Manager().dict()

    def __get_model_path(self, algorithm, hw, target):
        return os.path.join(self.models_path, f'{algorithm}_{hw}_{target}_DecisionTree_10')

    def get_model(self, algorithm, hw, target):
        """Returns the model (Decision).

        Args:
            algorithm (str): algorithm id.
            hw (str): hardware platform id
            target (str): target id.

        Raises:
            Exception: if model is not found and is already being trained.

        Returns:
            sklearn.tree.DecisionTreeRegressor: DT model.
        """
        model_path = self.__get_model_path(algorithm, hw, target)

        if not os.path.exists(model_path):
            if (algorithm, hw, target) in self.ongoing_training:
                raise Exception(f'Model for ({algorithm}, {hw}, {target}) training is ongoing. Come back later.')
            else:
                # launching training in background
                dataset = self.datasets.get_dataset(hw, )
                self.ongoing_training[(algorithm, hw, target)] = True
                p = Process(target=self.__run_training, args=(algorithm, 
                                                              hw,
                                                              target,
                                                              dataset))
                p.start()
                # raise FileNotFoundError(f'Model for ({algorithm}, {hw}, {target}) does not exist. Training started. Come back later.')
                # without the Exception, nothing is shown in the GUI, but multiple models can be trained in a single
                # request, while still keeping all the training part incapsulated in "get_model"
                print(f'Model for ({algorithm}, {hw}, {target}) does not exist. Training started.')
                p.join()
                del self.ongoing_training[(algorithm, hw, target)]
                print(f'Finished training model for ({algorithm}, {hw}, {target}).')

        # model exists, load it
        model = pickle.load(open(model_path, 'rb'))
        return model

    def __run_training(self, algorithm, hw, target, dataset):
        """
        Trains a Decision Tree and stores it with pickle.

        Args:
            algorithm (str): algorithm id.
            hw (str): hardware platform id.
            target (str): target id.
            dataset (pd.DataFrame): training dataset.
        
        """
        # s = time.time()
        model_path = self.__get_model_path(algorithm, hw, target)

        # filtering dataset for the specific hyperparams and target
        hyperparams = self.db.get_hyperparams()
        X = dataset[hyperparams].values  # Algorithm hyperparameters are our input values
        y = dataset[[target]].values

        # training the DT
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X, y)

        # storing the DT
        pickle.dump(dt, open(model_path, 'wb'))

        # print(self.ongoing_training)
        # print(f'Done in {time.time()-s}')
        # del self.ongoing_training[(algorithm, hw, target)]
        # print(self.ongoing_training)
