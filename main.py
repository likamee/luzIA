import config as cfg
import train
import evaluate as evl

def main(action):
    if action == 'train':
        train.process_train()
    elif action == 'evaluate':
        evl.evaluate()


main('train')











