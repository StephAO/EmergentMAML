from Tasks import referential_game

def main():
    epochs = 1000
    rg = referential_game.Referential_Game()
    losses = []
    for e in range(epochs):
        loss, acc = rg.play_game()
        if (e + 1) % 10 == 0:
            print(loss, acc)
        losses.append(loss)



if __name__ == "__main__":
    main()