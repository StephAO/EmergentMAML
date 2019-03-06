import sys
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
from Tasks import referential_game

def main(epochs=10000, D=1, K=100, use_images=True, loss_type='pairwise'):
    """
    Run epochs of games
    :return:
    """
    rg = referential_game.Referential_Game(K=K, D=D, use_images=use_images, 
        loss_type=loss_type)
    losses = []
    accuracies = []
    for e in range(1, epochs + 1):
        loss, acc = rg.play_game(e)

        # Print and collect stats
        if (e) % 20 == 0:
            print("loss: {0:1.4f}, accuracy: {1:3.2f}%".format(np.mean(losses[-20:]), np.mean(accuracies[-20:])*100))
        if e % 100 == 0:
            print("--- EPOCH {0:5d} ---".format(e))
        losses.append(loss)
        accuracies.append(acc)

        # 100% Success - end training
        if np.mean(accuracies[-20:]) == 1.0:
            break

    print("--- EPOCH {0:5d} ---".format(e))
    ml = max(losses)
    losses_ = [l / ml for l in losses]
    accuracies_ = [np.mean(accuracies[i: i + 10]) for i in range(len(accuracies) - 10)]
    
    plt.plot(losses_, 'r', accuracies_, 'g')  # , lrs, 'b')
    # plt.show()
    
    plt.savefig("./output_graph.png")

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
        
        
    