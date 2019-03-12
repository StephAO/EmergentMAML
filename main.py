import sys
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
from Tasks import referential_game, image_captioning

def converged(losses, precision=0.0001, prev_n=3):
    """
    Checks the last prev_n entries are in the precision threshold in a list of losses to see if it's converged
    :param losses [list]: losses to check on
    :param precision [float]: how similar the losses have to be to demonstrate convergence
    :param prev_n[int]: number of losses in a row which have to be in the precision threshold
    :return [Bool]: True if converged, else False
    """
    if len(losses) < prev_n:
        return False

    for i in range(len(losses) - prev_n, len(losses) - 1):
        if abs(losses[i] - losses[i + 1]) > precision:
            return False

    return True

def main(epochs=10, D=7, K=500, L=15, use_images=True, loss_type='pairwise'):
    """
    Run epochs of games
    :return:
    """
    # rg = referential_game.ReferentialGame(K=K, D=D, L=L, use_images=use_images, loss_type=loss_type)
    rg = image_captioning.ImageCaptioning(K=K, L=L)

    losses = []

    # Starting point
    # print("Validating epoch 0:")
    # accuracy, loss = rg.train_epoch(0, data_type="val")
    # print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")

    # Start training
    for e in range(1, epochs + 1):
        print("Training epoch {0}".format(e))
        accuracy, loss = rg.train_epoch(e, data_type="train")
        print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")
        print("Validating epoch {0}".format(e))
        accuracy, loss = rg.train_epoch(e, data_type="val")
        print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")
        losses.append(loss)

        # End training if 100% communication rate or convergence reached on loss
        if accuracy == 1.0 or converged(losses):
            break


def bool_type(val):
    """
    Return the corresponding value of val
    """
    if val.lower() in ('y', 'yes', 'true', 't'):
        return True
    else:
        return False

if __name__ == "__main__":
    
    # Default values
    epochs = 10000
    D = 1
    K = 100
    use_images = True
    loss_type = 'pairwise'
    
    # Extract options from command line 
    args_parser = ap.ArgumentParser()
    
    args_parser.add_argument('-D', default=D, type=int, 
        help="Number of distractor images", dest="D")
    args_parser.add_argument('-K', default=K, type=int, help="Vocabulary size",
        dest="K")
    args_parser.add_argument('-E', '--epochs', default=epochs, type=int, 
        help="Number of epochs", dest="E")
    args_parser.add_argument('-I', '--images', default=use_images, type=bool_type, 
        help="use images or one hot encoded vectors", dest="I")
    args_parser.add_argument('-L', '--loss-type', default=loss_type, 
        choices=['invMSE', 'pairwise', 'MSE'], dest="L")
    
    args = args_parser.parse_args()
    
    main()#epochs=args.E, D=args.D, K=args.K, use_images=args.I, loss_type=args.L)
        
        
    