import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animate


class Particle:
    def __init__(self, min_pos, max_pos, min_vel, max_vel, dim, loss):
        self.__pos = np.zeros(dim)
        self.__vel = np.zeros(dim)
        for i in range(dim):
            self.__pos[i] = np.random.uniform(min_pos[i], max_pos[i], (1,))
            self.__vel[i] = np.random.uniform(min_vel[i], max_vel[i], (1,))
        self.__bestPos = np.zeros(dim)
        self.__fitness = loss(self.__pos)

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_bestPos(self, value):
        self.__bestPos = value

    def get_bestPos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness(self, value):
        self.__fitness = value

    def get_fitness(self):
        return self.__fitness


class PSO:
    def __init__(self, dim, size, num_iter, min_pos, max_pos, min_vel, max_vel, tolerance, loss, C1=2, C2=2, W=1):
        # this loss function must be single input single output
        self.loss = loss
        self.dim, self.size = dim, size
        self.num_iter = num_iter
        self.tolerance = tolerance
        self.bestFitness = np.array([float('Inf')])
        self.C1, self.C2, self.W = C1, C2, W
        # verifications
        assert len(min_pos) == dim and len(max_pos) == dim, 'unmatched dim (position)'
        assert len(min_vel) == dim and len(max_vel) == dim, 'unmatched dim (velocity)'
        self.min_pos, self.max_pos = np.array(min_pos), np.array(max_pos)
        self.min_vel, self.max_vel = np.array(min_vel), np.array(max_vel)
        # outputs
        self.bestPosition = np.zeros(dim)
        self.fitnessList = np.empty(0)
        self.positionList = np.zeros((0, dim))
        # initialization of partical swarm
        self.particalList = [Particle(self.min_pos, self.max_pos, self.min_vel, self.max_vel, self.dim, loss)
                             for i in range(self.size)]

    def set_bestFitness(self, value):
        self.bestFitness = value

    def get_bestFitness(self):
        return self.bestFitness

    def set_bestPosition(self, value):
        self.bestPosition = value

    def get_bestPosition(self):
        return self.bestPosition

    def update_vel(self, particle):
        velocity = self.W * particle.get_vel() + \
                   self.C1 * np.random.rand() * (particle.get_bestPos() - particle.get_pos()) + \
                   self.C2 * np.random.rand() * (self.get_bestPosition() - particle.get_pos())
        velocity[velocity < self.min_vel] = self.min_vel[velocity < self.min_vel]
        velocity[velocity > self.max_vel] = self.max_vel[velocity > self.max_vel]
        particle.set_vel(velocity)

    def update_pos(self, particle):
        position = particle.get_pos() + particle.get_vel()
        position[position < self.min_pos] = self.min_pos[position < self.min_pos]
        position[position > self.max_pos] = self.max_pos[position > self.max_pos]
        particle.set_pos(position)
        temp = self.loss(particle.get_pos())
        if temp < particle.get_fitness():
            particle.set_fitness(temp)
            particle.set_bestPos(position)
        if temp < self.get_bestFitness():
            self.set_bestFitness(temp)
            self.set_bestPosition(position)

    def update_ndim(self):
        for i in range(self.num_iter):
            for particle in self.particalList:
                self.update_vel(particle)
                self.update_pos(particle)
            self.fitnessList = np.hstack((self.fitnessList, self.get_bestFitness()))
            self.positionList = np.vstack((self.positionList, self.get_bestPosition()))
            print('The best fitness of iteration %d: %.12f' % (i + 1, self.get_bestFitness()))
            if self.get_bestFitness() < self.tolerance:
                break
        return self.fitnessList, self.positionList, self.get_bestPosition()

    def image_1d(self, group=False, save=None):
        fig = plt.figure(figsize=(9, 6))
        gridx = np.linspace(self.min_pos[0], self.max_pos[0], 201)
        plotdata = np.zeros_like(gridx)
        for i in range(len(gridx)):
            plotdata[i] = self.loss(gridx[i])
        plt.plot(gridx, plotdata, 'k-')
        if group:
            points, = plt.plot(np.zeros(self.size), np.zeros(self.size), 'bo')
            plt.xlim((self.min_pos[0], self.max_pos[0]))

            def generator():
                while True:
                    plotdata = np.zeros((0, 2))
                    for particle in self.particalList:
                        self.update_vel(particle)
                        self.update_pos(particle)
                        plotdata = np.vstack((plotdata, np.stack((particle.get_pos(), particle.get_fitness())).T))
                    print('The best value of current iteration: %.12f' % (self.get_bestFitness()))
                    if self.get_bestFitness() < self.tolerance:
                        break
                    yield plotdata

            def updater(plotdata):
                points.set_xdata(plotdata[:, 0])
                points.set_ydata(plotdata[:, 1])

            ani = animate.FuncAnimation(fig, updater, generator, interval=300)
            plt.show()
            if save is not None:
                ani.save(save, writer='imagemagick', fps=4)

        else:
            points, = plt.plot(np.zeros(self.size), np.zeros(self.size), 'bo')
            plt.xlim((self.min_pos[0], self.max_pos[0]))

            def generator():
                while True:
                    for particle in self.particalList:
                        self.update_vel(particle)
                        self.update_pos(particle)
                    print('The best value of current iteration: %.12f' % (self.get_bestFitness()))
                    if self.get_bestFitness() < self.tolerance:
                        break
                    yield self.get_bestPosition(), self.get_bestFitness()

            def updater(plotdata):
                points.set_xdata(plotdata[0])
                points.set_ydata(plotdata[1])

            ani = animate.FuncAnimation(fig, updater, generator, interval=300)
            plt.show()
            if save is not None:
                ani.save(save, writer='imagemagick', fps=4)

    def image_2d(self, group=False, save=None):
        fig = plt.figure(figsize=(9/2.54, 6.75/2.54))
        gridx = np.linspace(self.min_pos[0], self.max_pos[0], 201)
        gridy = np.linspace(self.min_pos[1], self.max_pos[1], 201)
        meshx, meshy = np.meshgrid(gridx, gridy)
        plotdata = np.zeros_like(meshx)
        for i in range(meshx.shape[0]):
            for j in range(meshy.shape[0]):
                with torch.no_grad():
                    plotdata[i, j] = self.loss([meshx[i, j], meshy[i, j]])
        xmin, ymin = np.unravel_index(np.argmin(plotdata), plotdata.shape)
        plt.contourf(gridx, gridy, plotdata, 100)
        plt.plot(gridx[ymin], gridy[xmin], 'yo')
        plt.colorbar()
        if group:
            points, = plt.plot(np.zeros(self.size), np.zeros(self.size), 'wo')
            plt.xlim((self.min_pos[0], self.max_pos[0]))
            plt.ylim((self.min_pos[1], self.max_pos[1]))

            def generator():
                while True:
                    plotdata = np.zeros((0, 2))
                    for particle in self.particalList:
                        self.update_vel(particle)
                        self.update_pos(particle)
                        plotdata = np.vstack((plotdata, particle.get_pos()))
                    print('The best value of current iteration: %.12f' % (self.get_bestFitness()))
                    if self.get_bestFitness() < self.tolerance:
                        break
                    yield plotdata

            def updater(plotdata):
                points.set_xdata(plotdata[:, 0])
                points.set_ydata(plotdata[:, 1])

            ani = animate.FuncAnimation(fig, updater, generator, interval=300)
            plt.show()
            if save is not None:
                ani.save(save, writer='imagemagick', fps=4)
        else:
            points, = plt.plot(np.zeros(1), np.zeros(1), 'wo')
            plt.xlim((self.min_pos[0], self.max_pos[0]))
            plt.ylim((self.min_pos[1], self.max_pos[1]))

            def generator():
                while True:
                    for particle in self.particalList:
                        self.update_vel(particle)
                        self.update_pos(particle)
                    print('The best value of current iteration: %.12f' % (self.get_bestFitness()))
                    if self.get_bestFitness() < self.tolerance:
                        break
                    yield self.get_bestPosition()

            def updater(plotdata):
                points.set_xdata(plotdata[0])
                points.set_ydata(plotdata[1])

            ani = animate.FuncAnimation(fig, updater, generator, interval=300)
            plt.show()
            if save is not None:
                ani.save(save, writer='imagemagick', fps=4)
