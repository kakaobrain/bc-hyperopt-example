from train import Trainer


def main(args):
    t = Trainer()
    print(args)
    loss, accuracy = t.run(**args)
    return {
        'loss': loss,
        'accuracy': accuracy,
        'status': 'ok'
    }
