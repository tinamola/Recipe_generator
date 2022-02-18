from logging import getLogger
from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed
from logger import init_logger
from utils import data_preparation


def train(config):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    test_data, train_data, valid_data = data_preparation(config)
    model = Model(config).to(config['device'])
    trainer = Trainer(config, model)

    if config['test_only']:
        test_result = trainer.evaluate(test_data, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        logger.info('best valid loss: {}, best valid ppl: {}'.format(best_valid_score, best_valid_result))
        test_result = trainer.evaluate(test_data)

    logger.info('test result: {}'.format(test_result))


if __name__ == '__main__':
    config = Config(config_dict={'test_only': True,
                                 'load_experiment': 'saved/Fire-At-Oct-26-2021_07-05-47.pth'})
    train(config)
