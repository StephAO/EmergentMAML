# TODO use init.py to clean up imports
from Tasks import ReferentialGame, ImageCaptioning, Reptile, ImageSelection
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner, ImageSelector
from utils.data_handler import Data_Handler
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

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

def save_models(exp_key):
    SenderAgent.save_model(exp_key)
    ReceiverAgent.save_model(exp_key)

def main(epochs=55, task="reptile", D=127, K=10000, L=15, loss_type='pairwise'):
    """
    Run epochs of games
    :return:
    """
    load_key="22e65a1cdc6a4011a83ef0256b66ac4d"
    track_results=True

    Agent.set_params(K=K, D=D, L=L, loss_type=loss_type, train=True)
    dh = Data_Handler(batch_size=Agent.batch_size, same_category=False)

    with tf.variable_scope("all", reuse=tf.AUTO_REUSE):
        # Set up Agents and Tasks
        if task.lower() in ["rg", "referential game", "referential_game", "referentialgame"]:
            s = SenderAgent()
            r = ReceiverAgent(*s.get_output())
            s.all_agents_initialized(load_key)
            r.all_agents_initialized(load_key)
            t = ReferentialGame(s, r, data_handler=dh, track_results=track_results)
            dh.set_params(distractors=D )
        elif task.lower() in ["ic", "image captioning", "image_captioning", "imagecaptioning"]:
            ic = ImageCaptioner()
            ic.all_agents_initialized(load_key)
            t = ImageCaptioning(ic, data_handler=dh, track_results=track_results)
        elif task.lower() in ["is", "image selection", "image_selection", "imageselection"]:
            is_ = ImageSelector()
            is_.all_agents_initialized(load_key)
            t = ImageSelection(is_, data_handler=dh, track_results=track_results)
            dh.set_params(distractors=D)
        elif task.lower() in ["r", "reptile"]:
            t = Reptile(data_handler=dh, track_results=track_results, load_key=load_key)
        else:
            raise ValueError("Unknown task {}, select from ['referential_game', 'image_captioning']".format(task))

        # Initialize TF
        variables_to_initialize = tf.global_variables()
        if load_key is not None:
            dont_initialize = []
            if SenderAgent.loaded:
                dont_initialize += SenderAgent.get_all_weights()
            if ReceiverAgent.loaded:
                dont_initialize += ReceiverAgent.get_all_weights()
            variables_to_initialize = [v for v in tf.global_variables() if v not in dont_initialize]
        Agent.sess.run(tf.variables_initializer(variables_to_initialize))

        exp_key = t.get_experiment_key()
        losses = []
        best_accuracy = 0.0

        # Starting point
        if not isinstance(t, Reptile):
            print("Validating epoch 0:")
            accuracy, loss = t.train_epoch(0, mode="val")
            print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")

        # Start training
        for e in range(1, epochs + 1):
            print("Training epoch {0}".format(e))
            train_accuracy, train_loss = t.train_epoch(e, mode="train")
            print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(train_loss, train_accuracy * 100), end="\n")
            if not isinstance(t, Reptile):
                print("Validating epoch {0}".format(e))
                val_accuracy, val_loss = t.train_epoch(e, mode="val")
                print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(val_loss, val_accuracy * 100), end="\n")
                losses.append(val_loss)

                # End training if 100% communication rate or convergence reached on loss
                if val_accuracy == 1.0 or converged(losses):
                    break

                if val_accuracy > best_accuracy:
                    print("Saving Model")
                    best_accuracy = val_accuracy
                    save_models(exp_key)
            else:
                print("Saving Model")
                save_models(exp_key)

        Agent.sess.close()


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
        
        
    
