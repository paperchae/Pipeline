from abc import *


def build_model(model_config, device):
    print(f"Building {model_config.model_name} model...")


class TrainBase(metaclass=ABCMeta):
    def __init__(self, config, **kwargs):
        self.cfg = config
        self.model = build_model(self.cfg, kwargs['device'])

    @abstractmethod
    def run_epochs(self):
        raise NotImplementedError

    @abstractmethod
    # def train(self, loss, optimizer, scheduler):
    def train(self, loss, optimizer):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self, epoch):
        raise NotImplementedError
