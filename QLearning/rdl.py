import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from abc import ABC
import os.path
import matplotlib.pyplot as plt


class TicTacToe:
    def __init__(self):
        self.state = np.zeros(9)
        self.player1 = None
        self.player2 = None
        self.player = None

    def getState(self):
        if self.player == self.player1:
            turn = 1
        else:
            turn = 2
        return np.concatenate((np.asarray([turn]), self.state))

    def printBoard(self):
        print('\n -----')
        print('|' + str(self.state[0]) + '|' + str(self.state[1]) + '|' + str(self.state[2]) + '|')
        print(' -----')
        print('|' + str(self.state[3]) + '|' + str(self.state[4]) + '|' + str(self.state[5]) + '|')
        print(' -----')
        print('|' + str(self.state[6]) + '|' + str(self.state[7]) + '|' + str(self.state[8]) + '|')
        print(' -----\n')

    def play(self, choice):
        if self.player == self.player1:
            self.state[choice] = 1
        else:
            self.state[choice] = 2

    def changeTurn(self):
        if self.player == self.player1:
            self.player = self.player2
        else:
            self.player = self.player1

    def isDone(self):
        for x in range(0, 3):
            y = x * 3
            if self.state[y] != 0:
                if (self.state[y] == self.state[(y + 1)] and self.state[y] == self.state[(y + 2)]):
                    return 1
            if self.state[x] != 0:
                if (self.state[x] == self.state[(x + 3)] and self.state[x] == self.state[(x + 6)]):
                    return 1
        if ((self.state[0] != 0 and self.state[0] == self.state[4] and self.state[0] == self.state[8]) or
                (self.state[2] != 0 and self.state[2] == self.state[4] and self.state[4] == self.state[6])):
            return 1
        if not np.isin(0, self.state):
            return 2
        return 0

    def game(self, player1, player2):

        self.state = np.zeros(9)

        self.player1 = player1
        self.player2 = player2

        self.player1.reward = 0
        self.player2.reward = 0

        self.player = player1

        moves1 = []
        moves2 = []

        turnCount = 0

        while self.isDone() == 0:
            choice = self.player.play(self.getState())
            if self.state[choice] != 0:
                break
            if self.player == self.player1:
                moves1.append([np.asarray([self.getState()]), choice])
            else:
                moves2.append([np.asarray([self.getState()]), choice])
            turnCount += 1
            self.play(choice)
            if self.isDone() == 1:
                if self.player == self.player1:
                    print("Player 1 wins in " + str(turnCount) + " turns !")
                    break
                else:
                    print("Player 2 wins in " + str(turnCount) + " turns !")
                    break
            elif self.isDone() == 2:
                print("Draw !")
            self.changeTurn()
            if self.player1.isHuman or self.player2.isHuman:
                self.printBoard()

        if self.isDone() == 1:
            if self.player == self.player1:
                self.player1.reward = 10 - turnCount
                self.player2.reward = -5
            else:
                self.player1.reward = -5
                self.player2.reward = 10 - turnCount

        if self.isDone() != 0:
            if player1.isIA:
                player1.learn(moves1)
            if player2.isIA:
                player2.learn(moves2)


class Player(ABC):
    id = 1

    def __init__(self):
        self.isIA = False
        self.isRandom = False
        self.isHuman = False

        self.training = False

        self.eps = 0.25

        self.reward = 0

        self.model = None

    def play(self, state):
        pass


class IA(Player):
    def __init__(self):
        Player.__init__(self)

        self.id = Player.id
        Player.id += 1

        self.isIA = True
        self.losses = []

    def play(self, state):
        s = np.asarray([state])
        if self.training:
            if np.random.random() < self.eps:
                while 1:
                    choice = np.random.randint(0, 9)
                    if env.state[choice] == 0:
                        break
            else:
                choice = np.argmax(self.model.predict(s)[0])
        else:
            choice = np.argmax(self.model.predict(s)[0])
        if state[choice + 1] != 0:
            target = self.model.predict(s)[0]
            target[choice] = -100
            target = np.asarray([target])
            self.model.train_on_batch(s, target)
        return choice

    def save(self):
        self.model.save_weights("player" + str(self.id) + ".h5")

    def load(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='linear', input_shape=(10,)))
        self.model.add(Dense(9, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        if os.path.isfile("player" + str(self.id) + ".h5"):
            self.model.load_weights("player" + str(self.id) + ".h5")

    def learn(self, moves):
        if len(moves) > 0:
            for move in moves:
                target = self.model.predict(move[0])[0]
                target[move[1]] = self.reward
                target = np.asarray([target])
                loss = self.model.train_on_batch(move[0], target)
            self.losses.append(loss[0])
            print("N°" + str(self.id) + " [D loss: %f, reward: %.2f]" % (loss[0], self.reward))


class Human(Player):
    def __init__(self):
        Player.__init__(self)
        self.isHuman = True

    def play(self, state):
        while 1:
            try:
                choice = int(input(">> "))
            except:
                print("please enter a valid field")
                continue
            if state[choice] != 0:
                print("illegal move, plase try again")
                continue
            break
        return choice - 1


class Random(Player):
    def __init__(self):
        Player.__init__(self)
        self.isRandom = True

    def play(self, state):
        while 1:
            choice = np.random.randint(0, 9)
            if env.state[choice] == 0:
                break
        return choice


if __name__ == '__main__':
    env = TicTacToe()

    player1 = IA()
    player2 = IA()

    player1.load()
    player2.load()

    random = Random()
    human = Human()

    var = int(input("\nWhat to do ?\n1- Train\n2- Play"))
    if var == 1:
        var = int(input("\nNumber of games : "))
        player1.training = True
        player2.training = True
        for i in range(var):
            print("\nGame n°" + str(i))
            env.game(player1, player2)
        plt.plot(player1.losses)
        plt.show()
        plt.plot(player2.losses)
        plt.show()
    else:
        var = int(input("\nWho are you ?\n1- Player 1\n2- Player 2"))
        if var == 1:
            env.game(human, player2)
        else:
            env.game(player1, human)

    player1.save()
    player2.save()
