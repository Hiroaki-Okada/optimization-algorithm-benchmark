import math
import random

import numpy as np

# evaluate メソッドや generate_initial_point メソッドはインスタンスに依存しない
# そのため、通常のインスタンスメソッドではなく静的メソッドまたはクラスメソッドとして実装して良い
# 静的メソッドはインスタンス化せずに呼び出せるメソッドで、インスタンスに依存しない単純な関数やヘルパーメソッドの実装に使用される
# クラスメソッドもインスタンス化せずに呼び出すことができ、クラス全体に関連する処理の実装に用いられる
# 今回は静的メソッドで実装した


class Sphere:
    min_x, max_x = -50.0, 50.0
    min_y, max_y = -50.0, 50.0
    min_f, max_f = 1e-8, 5000
    levs = np.geomspace(min_f, max_f, num=50)
    name = 'sphere'

    @staticmethod
    def evaluate(x, y):
        return x ** 2 + y ** 2
    
    @staticmethod
    def generate_initial_point(seed=0):
        random.seed(seed)
        x = random.uniform(Sphere.min_x, Sphere.max_x)
        y = random.uniform(Sphere.min_y, Sphere.max_y)

        return x, y

    @staticmethod
    def generate_random_point():
        x = random.uniform(Sphere.min_x, Sphere.max_x)
        y = random.uniform(Sphere.min_y, Sphere.max_y)

        return x, y


class Rosenblock:
    min_x, max_x = -2.0, 2.0
    min_y, max_y = -1.0, 3.0
    min_f, max_f = 1e-8, 2500
    levs = np.geomspace(min_f, max_f, num=50)
    name = 'rosenbrock'

    @staticmethod
    def evaluate(x, y, a=1, b=100):
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    @staticmethod
    def generate_initial_point(seed=4):
        random.seed(seed)
        x = random.uniform(Rosenblock.min_x, Rosenblock.max_x)
        y = random.uniform(Rosenblock.min_y, Rosenblock.max_y)

        return x, y

    @staticmethod
    def generate_random_point():
        x = random.uniform(Rosenblock.min_x, Rosenblock.max_x)
        y = random.uniform(Rosenblock.min_y, Rosenblock.max_y)

        return x, y


class Beale:
    min_x, max_x = -4.0, 4.0
    min_y, max_y = -4.0, 4.0
    min_f, max_f = 1e-8, 70000
    levs = np.geomspace(min_f, max_f, num=50)
    name = 'beale'

    @staticmethod
    def evaluate(x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2
    
    @staticmethod
    def generate_initial_point(seed=4):
        random.seed(seed)
        x = random.uniform(Beale.min_x, Beale.max_x)
        y = random.uniform(Beale.min_y, Beale.max_y)

        return x, y

    @staticmethod
    def generate_random_point():
        x = random.uniform(Beale.min_x, Beale.max_x)
        y = random.uniform(Beale.min_y, Beale.max_y)

        return x, y    


class ThreeHumpCamel:
    min_x, max_x = -2.0, 2.0
    min_y, max_y = -2.0, 2.0
    min_f, max_f = 1e-8, 10
    levs = np.geomspace(min_f, max_f, num=50)
    name = 'three-hump_camel'

    @staticmethod
    def evaluate(x, y):
        return 2 * (x ** 2) - 1.05 * (x ** 4) + (x ** 6) / 6 + x * y + y ** 2

    @staticmethod
    def generate_initial_point(seed=4):
        random.seed(seed)
        x = random.uniform(ThreeHumpCamel.min_x, ThreeHumpCamel.max_x)
        y = random.uniform(ThreeHumpCamel.min_y, ThreeHumpCamel.max_y)

        return x, y

    @staticmethod
    def generate_random_point():
        x = random.uniform(ThreeHumpCamel.min_x, ThreeHumpCamel.max_x)
        y = random.uniform(ThreeHumpCamel.min_y, ThreeHumpCamel.max_y)

        return x, y


class Himmelblau:
    min_x, max_x = -6.0, 6.0
    min_y, max_y = -6.0, 6.0
    min_f, max_f = 1e-8, 2000
    levs = np.geomspace(min_f, max_f, num=50)
    name = 'himmelblau'

    @staticmethod
    def evaluate(x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def generate_initial_point(seed=9):
        random.seed(seed)
        x = random.uniform(Himmelblau.min_x, Himmelblau.max_x)
        y = random.uniform(Himmelblau.min_y, Himmelblau.max_y)

        return x, y

    @staticmethod
    def generate_random_point():
        x = random.uniform(Himmelblau.min_x, Himmelblau.max_x)
        y = random.uniform(Himmelblau.min_y, Himmelblau.max_y)

        return x, y


class MullerBrownPotential:
    min_x, max_x = -2.5, 1.5
    min_y, max_y = -1.0, 3.0
    min_f, max_f = -150, 300
    levs = np.arange(min_f, max_f, 10)
    name = 'muller_brown_potential'

    @staticmethod
    def evaluate(x, y):
        A_i = [-200, -100, -170, 15]
        a_i = [-1, -1, -6.5, 0.7]
        b_i = [0, 0, 11, 0.6]
        c_i = [-10, -10, -6.5, 0.7]
        X_i = [1, 0, -0.5, -1]
        Y_i = [0, 0.5, 1.5, 1.0]

        energy = 0
        for i in range(4):
            term_1 = a_i[i] * (x - X_i[i]) ** 2
            term_2 = b_i[i] * (x - X_i[i]) * (y - Y_i[i])
            term_3 = c_i[i] * (y - Y_i[i]) ** 2
            energy += A_i[i] * math.exp(term_1 + term_2 + term_3)

        return energy
    
    @staticmethod
    def generate_initial_point(seed=1):
        random.seed(seed)
        x = random.uniform(MullerBrownPotential.min_x, MullerBrownPotential.max_x)
        y = random.uniform(MullerBrownPotential.min_y, MullerBrownPotential.max_y)

        return x, y
    
    @staticmethod
    def generate_random_point():
        x = random.uniform(MullerBrownPotential.min_x, MullerBrownPotential.max_x)
        y = random.uniform(MullerBrownPotential.min_y, MullerBrownPotential.max_y)

        return x, y