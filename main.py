# TODO use init.py to clean up imports
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner
from Tasks import ReferentialGame, ImageCaptioning
from utils.data_handler import Data_Handler
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import sys

import tensorflow as tf

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

def main(epochs=10, task="rg", D=15, K=500, L=1, use_images=True, loss_type='pairwise'):
    """
    Run epochs of games
    :return:
    """
    Agent.set_params(K=K, D=D, L=L, loss_type=loss_type)
    dh = Data_Handler(batch_size=Agent.batch_size, group=False)

    if task.lower() in ["rg", "referential game", "referential_game", "referentialgame"]:
        s = SenderAgent()
        r = ReceiverAgent(*s.get_output())
        s.set_loss(r.get_output()[1])
        t = ReferentialGame(s, r, dh)
        dh.set_params(images_per_instance=D + 1)
    elif task.lower() in ["ic", "image captioning", "image_captioning", "imagecaptioning"]:
        ic = ImageCaptioner()
        t = ImageCaptioning(ic, dh)
    else:
        raise ValueError("Unknown task {}, select from ['referential_game', 'image_captioning']".format(task))


    losses = []

    # Starting point
    print("Validating epoch 0:")
    accuracy, loss = t.train_epoch(0, mode="val")
    print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")

    summ_writer = tf.summary.FileWriter('/home/stephane/PycharmProjects/EmergentMAML/summaries/', Agent.sess.graph)

    # Start training
    for e in range(1, epochs + 1):
        print("Training epoch {0}".format(e))
        accuracy, loss = t.train_epoch(e, mode="train")
        print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")
        print("Validating epoch {0}".format(e))
        accuracy, loss = t.train_epoch(e, mode="val")
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
        
        
    