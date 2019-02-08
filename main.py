import matplotlib.pyplot as plt
from Tasks import referential_game

def main():
    epochs = 10000
    rg = referential_game.Referential_Game()
    losses = []
    accuracies = []
    for e in range(epochs):
        loss, acc = rg.play_game()
        if (e + 1) % 10 == 0:
            print(loss, acc)
        losses.append(loss)
        accuracies.append(acc)

    ml = max(losses)
    losses = [l / ml for l in losses]
    plt.plot(losses, 'r', accuracies, 'g')
    plt.show()



if __name__ == "__main__":
    main()