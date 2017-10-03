import random
from math import e, pow


#[inputs, hc_1, hc_2, ..., outputs]

def sigmoid(value, derivative=False):

    if derivative:
        x = sigmoid(value)
        return x * (1 - x)
    else:
        return 1 / (1 + pow(e, -value))

def relu(value, derivative=False):

    if derivative:
        return 1 if value > 0 else 0
    else:
        return value if value > 0 else 0

# [1, 0, 1] [0, 1, 1]
def distance(values, expected):

    count = 0
    total = 0

    for i in range(len(values)):
        total += abs(values[i] - expected[i])
        count += 1

    return total/count

class Net:

    def __init__(self, layers, activation, deviation=10):

        prev_count = layers[0] + 1
        self.activation = activation

        net = []

        for count in layers[1:]:
            layer = []
            for i in range(count):
                neuron = [random.uniform(-deviation, deviation) for j in range(prev_count)]
                layer.append(neuron)
            prev_count = count + 1
            net.append(layer)

        self.net_weights = net

    def print(self):
        pass

    def forward_propagation(self, nums):

        outs = [1] + nums

        for layer in self.net_weights:

            out_new = []

            for neuron in layer:

                total = 0

                for i, weight in enumerate(neuron):

                    total += outs[i] * weight

                out_new.append(self.activation(total))
            outs = [1] + out_new

        return outs[1:]

    def average_distance(self, inputs, outputs):

        total_distance = 0
        count = 0

        for i in range(len(inputs)):
            values = self.forward_propagation(inputs[i])

            difference = distance(values, outputs[i])

            total_distance += difference
            count += 1

        avg_distance = total_distance / count
        return avg_distance

    def vector(self):

        vector = []

        for layer in self.net_weights:
            for neuron in layer:
                for weight in neuron:
                    vector.append(weight)

        return vector

    def update_weights(self, new_weights):

        position = 0

        for layer in self.net_weights:
            for neuron in layer:
                for i, weight in enumerate(neuron):
                    neuron[i] = new_weights[position]
                    position += 1

class Swarm:

    def __init__(self, layers, activation, count=5, deviation=5, learning_rate=.1):
        self.nets = []
        self.layers = layers
        self.count = count
        self.deviation = deviation
        self.learning_rate = learning_rate
        self.activation = activation

        for i in range(count):

            self.nets.append(Net(layers, activation, deviation))


    def add_net(self):
        self.nets.append(Net(self.layers, self.activation, self.deviation))


    def find_best(self, inputs, outputs):
        best_avg_difference = float("inf")
        best_net = self.nets[0]

        for net in self.nets:

            avg_distance = net.average_distance(inputs, outputs)

            if avg_distance < best_avg_difference:
                best_avg_difference = avg_distance
                best_net = net

        return best_net

    def train(self, inputs, outputs, epochs=100, threshold=.5, report_rate=100):

        converged = False

        for epoch in range(epochs):

            if epoch % report_rate == 0:
                print("Currently on epoch:", epoch, "Has converged since last report:", converged)
                converged = False

            self.best = self.find_best(inputs, outputs)

            for net in self.nets:

                self.move_towards_best(net)

            if self.converged(threshold):
                #print('The swarm has converged. Resetting.')

                converged = True

                self.nets = [self.best]

                while len(self.nets) < self.count:
                    self.add_net()


    def converged(self, threshold):

        count = 0
        total_distance = 0

        for net in self.nets:

            difference = distance(self.best.vector(), net.vector())
            total_distance += difference
            count += 1

        avg_distance = total_distance / count

        return avg_distance < threshold



    def move_towards_best(self, net):

        best_vector = self.best.vector()
        vector = net.vector()

        new_weights = []

        for i, value in enumerate(vector):

            delta = (value - best_vector[i]) * self.learning_rate
            weight = value - delta

            new_weights.append(weight)


        net.update_weights(new_weights)

    def test(self, inputs, outputs):

        total_distance = 0
        count = 0

        for i in range(len(inputs)):

            total_distance += distance(self.best.forward_propagation(inputs[i]), outputs[i])
            count += 1

        return total_distance/count


if __name__ == '__main__':
    random.seed(1)

    layers = [2, 1, 1]


    swarm = Swarm(layers, relu, deviation=5, count=5, learning_rate=.1)

    inputs = [[1, 1], [0, 1], [1, 0], [0, 0]]
    outputs = [[0], [1], [1], [1]]

    swarm.train(inputs, outputs, 10000, threshold=.1)


    print("Out:", swarm.best.forward_propagation(inputs[0]), "Expected:", outputs[0])
    print("Out:", swarm.best.forward_propagation(inputs[1]), "Expected:", outputs[1])
    print("Out:", swarm.best.forward_propagation(inputs[2]), "Expected:", outputs[2])
    print("Out:", swarm.best.forward_propagation(inputs[3]), "Expected:", outputs[3])

    print(swarm.best.vector())