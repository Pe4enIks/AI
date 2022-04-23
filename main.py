from utils.dataset import get_dataset
from utils.config import parse_config
from utils.models import SVM
from sklearn.metrics import f1_score, precision_score, recall_score


def main(config_file):
    config = parse_config(config_file)

    train, test = get_dataset(dataset=config['dataset'], test_size=config['test_size'], random_state=config['random_state'])
    model = SVM(lr=config['lr'], alpha=config['alpha'], epochs=config['epochs'])

    model.fit(train=train, test=test)
    pred = model.predict(test[0])

    pr = precision_score(test[1], pred)
    re = recall_score(test[1], pred)
    f1 = f1_score(test[1], pred)
    print(f'precision={pr}, recall={re}, f1={f1}')


if __name__ == '__main__':
    config_file = './config.yaml'
    main(config_file)
