import matplotlib.pyplot as plt
import numpy as np
from Tasks import referential_game

def main():
    """
    Run epochs of games
    :return:
    """
    epochs = 10000
    rg = referential_game.Referential_Game()
    losses = []
    accuracies = []
    for e in range(1, epochs + 1):
        loss, acc = rg.play_game(e)

        # Print and collect stats
        if (e) % 20 == 0:
            print(loss, np.mean(accuracies[-20:]))
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



if __name__ == "__main__":
    main()