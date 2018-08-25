import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from abc import ABC
import os.path
import matplotlib.pyplot as plt


class TicTacToe:
    def __init__(self):
        self.state = np.zeros(9)

    def getInput(self):
        input = []
        for s in self.state:
            vector = [0] * 3
            vector[int(s)] = 1
            input += vector
        return input

    def printBoard(self):
        print('\n -----')
        print('|' + str(self.state[0]) + '|' + str(self.state[1]) + '|' + str(self.state[2]) + '|')
        print(' -----')
        print('|' + str(self.state[3]) + '|' + str(self.state[4]) + '|' + str(self.state[5]) + '|')
        print(' -----')
        print('|' + str(self.state[6]) + '|' + str(self.state[7]) + '|' + str(self.state[8]) + '|')
        print(' -----\n')

    def play(self, choice, bool):
        if bool:
            self.state[choice - 1] = 1
        else:
            self.state[choice - 1] = 2

    def isDone(self):
        for x in range(0, 3):
            y = x * 3
            if self.state[y] != 0:
                if self.state[y] == self.state[y + 1] and self.state[y] == self.state[y + 2]:
                    return 1
            if self.state[x] != 0:
                if self.state[x] == self.state[x + 3] and self.state[x] == self.state[x + 6]:
                    return 1
        if ((self.state[0] != 0 and self.state[0] == self.state[4] and self.state[0] == self.state[8]) or
                (self.state[2] != 0 and self.state[2] == self.state[4] and self.state[4] == self.state[6])):
            return 1
        if not np.isin(0, self.state):
            return 2
        return 0

    def game(self, player1, player2):

        self.state = np.zeros(9)

        player = player2

        player1.reward = 0
        player1.moves = []

        count = 0

        while self.isDone() == 0:
            if player == player1:
                player = player2
            else:
                player = player1
            choice = player.play()
            if self.state[choice - 1] != 0:
                print("Illegal move, shut down")
                return -1
            if player == player1:
                player1.moves.append([np.asarray([self.getInput()]), choice, True])
            self.play(choice, player == player1)
            count += 1
            if self.isDone() == 1:
                if player == player1:
                    print("Player 1 wins in " + str(count) + " turns !")
                    player1.reward = 1
                    break
                else:
                    print("Player 2 wins in " + str(count) + " turns !")
                    player1.reward = -1
                    break
            elif self.isDone() == 2:
                print("Draw !")
                return 0
            if player1.kind == 2 or player2.kind == 2:
                self.printBoard()
        if player1.kind == 1:
            player1.learn()
        if player == player1:
            return 1
        else:
            return 2


class Player(ABC):
    id = 1

    def __init__(self):
        self.kind = 0
        self.reward = 0
        self.moves = []

    def play(self):
        pass


class IA(Player):
    def __init__(self):
        Player.__init__(self)
        self.id = Player.id
        self.kind = 1
        self.losses = []
        self.model = None
        self.training = False
        Player.id += 1

    def play(self):
        state = env.state
        s = np.asarray([env.getInput()])
        if self.training:
            if np.random.random() < 0.25:
                while 1:
                    choice = np.random.randint(1, 10)
                    if state[choice - 1] == 0:
                        break
            else:
                choice = np.argmax(self.model.predict(s)[0]) + 1
        else:
            choice = np.argmax(self.model.predict(s)[0]) + 1
        if state[choice - 1] != 0:
            target = self.model.predict(s)[0]
            target[choice - 1] = -100
            target = np.asarray([target])
            self.model.train_on_batch(s, target)
        return choice

    def save(self):
        self.model.save_weights("player" + str(self.id) + ".h5")

    def load(self):
        self.model = Sequential()
        self.model.add(Dense(9, activation='sigmoid', input_shape=(27,)))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        if os.path.isfile("player" + str(self.id) + ".h5"):
            self.model.load_weights("player" + str(self.id) + ".h5")

    def learn(self):
        if len(self.moves) > 0 and self.training:
            for move in self.moves:
                env.state = move[0][0]
                target = self.model.predict(move[0])[0]
                env.play(move[1], move[2])
                target[move[1] - 1] = self.reward
                target = np.asarray([target])
                loss = self.model.train_on_batch(move[0], target)
            self.losses.append(loss[0])
            print("[D loss: %f, reward: %.2f]" % (loss[0], self.reward))


class Human(Player):
    def __init__(self):
        Player.__init__(self)
        self.kind = 2

    def play(self):
        state = env.state
        while 1:
            try:
                choice = int(input(">> "))
            except:
                print("Please enter a valid field")
                continue
            if state[choice - 1] != 0:
                print("Illegal move, plase try again")
                continue
            break
        return choice


class Random(Player):
    def __init__(self):
        Player.__init__(self)
        self.kind = 3

    def play(self):
        state = env.state
        while 1:
            choice = np.random.randint(1, 10)
            if state[choice - 1] == 0:
                break
        return choice


if __name__ == '__main__':
    env = TicTacToe()

    player1 = IA()
    player1.load()

    random = Random()
    human = Human()

    var = int(input("\nWhat to do ?\n1- Train\n2- Evaluate\n3- Play"))
    if var == 1:
        var = int(input("\nNumber of games : "))
        player1.training = True
        for i in range(var):
            print("\nGame n°" + str(i))
            env.game(player1, random)
        plt.plot(player1.losses)
        plt.show()
    elif var == 3:
        env.game(player1, human)
    else:
        count = 0
        for i in range(100):
            if env.game(player1, random) == 1:
                count += 1
        print(count / 100)

    player1.save()

