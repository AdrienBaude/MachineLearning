import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from abc import ABC
import os.path
import matplotlib.pyplot as plt


class TicTacToe:
    def __init__(self):
        self.state = np.zeros(9)
        self.player1 = None
        self.player2 = None
        self.player = None

    def reset(self):
        self.state = np.zeros(9)

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
        self.reset()

        self.player1 = player1
        self.player2 = player2

        self.player1.reward = 0
        self.player2.reward = 0

        self.player = player1

        moves1 = []
        moves2 = []

        turnCount = 0

        while self.isDone() == 0:
            self.player1.eps *= 0.999
            self.player2.eps *= 0.999

            choice = self.player.play(self.getState())

            if self.player == self.player1:
                moves1.append([np.asarray([self.getState()]), choice])
            else:
                moves2.append([np.asarray([self.getState()]), choice])
            turnCount += 1
            self.play(choice)
            if self.player1.isHuman or self.player2.isHuman:
                self.printBoard()
            if self.isDone() == 1:
                if self.player == self.player1:
                    print("Player 1 wins !")
                    break
                else:
                    print("Player 2 wins !")
                    break
            elif self.isDone() == 2:
                print("Draw !")
            self.changeTurn()

        if self.isDone() == 1:
            if self.player == self.player1:
                self.player1.reward = 10 - turnCount
                self.player2.reward = -10
            else:
                self.player1.reward = -10
                self.player2.reward = 10 - turnCount

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

        self.eps = 0.5

        self.reward = 0

    def play(self, state):
        pass


class IA(Player):
    def __init__(self):
        Player.__init__(self)

        self.model = Sequential()
        self.model.add(Dense(32, activation='linear', input_shape=(10,)))
        self.model.add(Dense(9, activation='linear'))
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        self.id = Player.id
        Player.id += 1

        self.isIA = True
        self.losses = []

    def play(self, state):
        s = np.asarray([state])
        count = 0
        while 1:
            if self.training:
                if np.random.random() < self.eps or count > 5:
                    while 1:
                        choice = np.random.randint(0, 9)
                        if state[choice + 1] == 0:
                            break
                else:
                    choice = np.argmax(self.model.predict(s)[0])
            else:
                choice = np.argmax(self.model.predict(s)[0])
            if state[choice + 1] != 0:
                target = self.model.predict(s)[0]
                target[choice] = -100
                target = np.asarray([target])
                loss = self.model.train_on_batch(s, target)
                self.losses.append(loss[0])
                if self.id == 1 or self.id == 2:
                    print("[D loss: %f, reward: -10.00]" % (loss[0]))
            else:
                break
            count += 1
        return choice

    def save(self):
        self.model.save_weights("player" + str(self.id) + ".h5")

    def load(self):
        if os.path.isfile("player" + str(self.id) + ".h5"):
            self.model.load_weights("player" + str(self.id) + ".h5")

    def learn(self, moves):
        for move in moves:
            target = self.model.predict(move[0])[0]
            target[move[1]] = self.reward
            target = np.asarray([target])
            loss = self.model.train_on_batch(move[0], target)
            self.losses.append(loss[0])
            if self.id == 1 or self.id == 2:
                print("[D loss: %f, reward: %.2f]" % (loss[0], self.reward))


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
            if state[choice + 1] == 0:
                break
        return choice


if __name__ == '__main__':
    env = TicTacToe()

    player1 = IA()
    player2 = IA()

    player1_1 = IA()
    player1_2 = IA()

    player2_1 = IA()
    player2_2 = IA()

    player1.load()
    player2.load()
    player1_1.load()
    player1_2.load()
    player2_1.load()
    player2_2.load()

    random = Random()

    human = Human()

    player1Trainers = [player1_1, player1_2, random]
    player2Trainers = [player2_1, player2_2, random]

    var = int(input("\nWhat to do ?\n1- Train\n2- Play"))
    if var == 1:
        var = int(input("\nWho will train ?\n1- Player 1\n2- Player 2"))
        var2 = int(input("\nNumber of games : "))
        if var == 1:
            player1.training = True
            for ia in player1Trainers:
                ia.training = True
            for i in range(var2):
                print("\nGame n°"+str(i))
                rand = np.random.randint(0, 3)
                env.game(player1, player1Trainers[rand])
            plt.plot(player1.losses)
            plt.show()
        else:
            player2.training = True
            for ia in player2Trainers:
                ia.training = True
            for i in range(var2):
                print("\nGame n°"+str(i))
                rand = np.random.randint(0, 3)
                env.game(player2, player2Trainers[rand])
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
    player1_1.save()
    player1_2.save()
    player2_1.save()
    player2_2.save()
