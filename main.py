# TODO use init.py to clean up imports
from Tasks import ReferentialGame, ImageCaptioning, Reptile, ImageSelection
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner, ImageSelector
from utils.data_handler import Data_Handler
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
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

def save_models(exp_key, sender=True, receiver=True):
    # Agent.save_model(exp_key)
    if sender:
        SenderAgent.save_model(exp_key)
    if receiver:
        ReceiverAgent.save_model(exp_key)

def main(epochs=50, task="ic", D=31, K=10000, L=10, use_images=True, loss_type='pairwise'):
    """
    Run epochs of games
    :return:
    """
    load_key=None#"23c353ddf0e545ccbd86ad7babeaf09e"

    Agent.set_params(K=K, D=D, L=L, loss_type=loss_type)
    dh = Data_Handler(batch_size=Agent.batch_size, group=False)

    s, r = False, False

    if task.lower() in ["rg", "referential game", "referential_game", "referentialgame"]:
        s = SenderAgent(load_key=load_key)
        r = ReceiverAgent(*s.get_output(), load_key=load_key)
        t = ReferentialGame(s, r, dh)
        dh.set_params(images_per_instance=D + 1)
        s, r = True, True
    elif task.lower() in ["ic", "image captioning", "image_captioning", "imagecaptioning"]:
        ic = ImageCaptioner(load_key=load_key)
        t = ImageCaptioning(ic, dh)
        s = True
    elif task.lower() in ["is", "image selection", "image_selection", "imageselection"]:
        is_ = ImageSelector(load_key=load_key)
        t = ImageSelection(is_, dh)
        dh.set_params(images_per_instance=D + 1)
        r = True
    elif task.lower() in ["r", "reptile"]:
        t = Reptile(dh)
        s, r = True
    else:
        raise ValueError("Unknown task {}, select from ['referential_game', 'image_captioning']".format(task))

    # Initialize TF
    variables_to_initialize = tf.global_variables()
    if load_key is not None:
        dont_initialize = []
        if s:
            dont_initialize += SenderAgent.get_all_weights()
        if r:
            dont_initialize += ReceiverAgent.get_all_weights()
        variables_to_initialize = [v for v in tf.global_variables() if v not in dont_initialize]

    Agent.sess.run(tf.variables_initializer(variables_to_initialize))

    print(len(tf.trainable_variables()))

    exp_key = t.get_experiment_key()
    losses = []

    # Starting point
    # print("Validating epoch 0:")
    # accuracy, loss = t.train_epoch(0, mode="val")
    # print("\rloss: {0:1.4f}, accuracy: {1:5.2f}%".format(loss, accuracy * 100), end="\n")

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

        save_models(exp_key, sender=s, receiver=r)

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
        
        
    
