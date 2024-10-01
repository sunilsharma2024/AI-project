import numpy as np

class RedFoxOptimizer:
    def __init__(self, func, bounds, num_foxes=30, iterations=100):
        self.func = func
        self.bounds = bounds
        self.num_foxes = num_foxes
        self.iterations = iterations

    def optimize(self):
        fox_positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_foxes, len(self.bounds[0])))
        best_position = fox_positions[0]
        best_value = self.func(best_position)

        for _ in range(self.iterations):
            for i in range(self.num_foxes):
                current_value = self.func(fox_positions[i])
                if current_value < best_value:
                    best_value = current_value
                    best_position = fox_positions[i]

            for i in range(self.num_foxes):
                # Update fox positions
                alpha = np.random.rand()
                beta = np.random.rand()
                fox_positions[i] += alpha * (best_position - fox_positions[i]) + beta * np.random.randn(len(self.bounds[0]))

                # Ensure positions are within bounds
                fox_positions[i] = np.clip(fox_positions[i], self.bounds[0], self.bounds[1])

        return best_position, best_value


class SeagullOptimizer:
    def __init__(self, func, bounds, num_seagulls=30, iterations=100):
        self.func = func
        self.bounds = bounds
        self.num_seagulls = num_seagulls
        self.iterations = iterations

    def optimize(self):
        seagull_positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_seagulls, len(self.bounds[0])))
        best_position = seagull_positions[0]
        best_value = self.func(best_position)

        for _ in range(self.iterations):
            for i in range(self.num_seagulls):
                current_value = self.func(seagull_positions[i])
                if current_value < best_value:
                    best_value = current_value
                    best_position = seagull_positions[i]

            for i in range(self.num_seagulls):
                # Update seagull positions
                direction = np.random.uniform(-1, 1, len(self.bounds[0]))
                seagull_positions[i] += direction * (best_position - seagull_positions[i])

                # Ensure positions are within bounds
                seagull_positions[i] = np.clip(seagull_positions[i], self.bounds[0], self.bounds[1])

        return best_position, best_value


class HybridRedFoxSeagullOptimizer:
    def __init__(self, func, bounds, num_foxes=15, num_seagulls=15, iterations=100):
        self.func = func
        self.bounds = bounds
        self.num_foxes = num_foxes
        self.num_seagulls = num_seagulls
        self.iterations = iterations

    def optimize(self):
        fox_optimizer = RedFoxOptimizer(self.func, self.bounds, self.num_foxes, self.iterations)
        seagull_optimizer = SeagullOptimizer(self.func, self.bounds, self.num_seagulls, self.iterations)

        best_fox_position, best_fox_value = fox_optimizer.optimize()
        best_seagull_position, best_seagull_value = seagull_optimizer.optimize()

        # Choose the best position from both optimizers
        if best_fox_value < best_seagull_value:
            return best_fox_position, best_fox_value
        else:
            return best_seagull_position, best_seagull_value


def HRFSG(func, lb, ub, dim, pop_size, max_iter):
    # Define bounds for the optimization
    bounds = [np.array(lb), np.array(ub)]

    # Create hybrid optimizer instance
    hybrid_optimizer = HybridRedFoxSeagullOptimizer(func, bounds, num_foxes=pop_size//2, num_seagulls=pop_size//2, iterations=max_iter)

    # Perform optimization
    best_position, best_value = hybrid_optimizer.optimize()

    return best_position, best_value




