import numpy as np
import cvxpy as cv
from typing import List
from barrier_function import BarrierFunction

class ControlBarrierFunction:

    def __init__(self, alpha: float):
        self.u = cv.Variable(2) # Velocity decision variable
        self.relaxation = cv.Variable(1)
        self.alpha = alpha
        self.h : List[BarrierFunction] = []
        self.theta = 0.0

    def add_constraint(self, barrier_function: BarrierFunction):
        if not (issubclass(type(barrier_function), BarrierFunction)):
            raise TypeError("Invalid constrain type. Must be a subclass of BarrierFunction.")

        self.h.append(barrier_function)


    def add_rotation(self, theta: float):
        self.theta = theta

    def evaluate(self, velocity):
        if self.h is None:
            return velocity

        objective = cv.Minimize(cv.sum_squares(self.u - velocity) + 10 * cv.square(self.relaxation))
        constraints = []
        try:
            for h in self.h:
                grad = h.gradient()
                a = grad[0] * np.cos(self.theta) + grad[1] * np.sin(self.theta)
                print(h.evaluate())
                constrain = a * self.u[0] + grad[2] * self.u[1] >= -self.alpha * h.evaluate()
                #constrain = a * self.u[0]  >= -self.alpha * h.evaluate()
                constraints.append(constrain)
        except Exception as e:
            print("Error in CBF evaluation:", e)
            return [0.0, 0.0]
        
        problem = cv.Problem(objective, constraints)
        problem.solve(solver=cv.OSQP)

        return self.u.value