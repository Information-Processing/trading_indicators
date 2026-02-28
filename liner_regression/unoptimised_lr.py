import numpy as np
import time

class UnoptimisedSoftwareLR:
    def __init__(self, initial_data, collumn_headers):
        self.collumn_headers = collumn_headers
        self.num_params = len(initial_data[0])
        
        #start with initial data
        self.a = np.concatenate([initial_data[:, :-1], np.ones((len(initial_data), 1))], axis=1)
        self.b = initial_data[:, -1].reshape(-1, 1)

        self.params = self.solve()

    def solve(self):
        #solve function for doing (ATA)^-1 * (ATB)
        ata = self.a.T @ self.a
        atb = self.a.T @ self.b
        ata_inv = np.linalg.inv(ata)
        return ata_inv @ atb

    def print_equation(self):
        params = [p.item() for p in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="")
        for i in range(1, self.num_params):
            print(f" + {params[i]:.2f} * {self.collumn_headers[i]}", end="")
        print("\n")

    def stream_chunk(self, lines):
        #stream chunk resolves/retrains the linear regression equation every time we add samples to it
        new_a = np.concatenate([lines[:, :-1], np.ones((len(lines), 1))], axis=1)
        new_b = lines[:, -1].reshape(-1, 1)

        self.a = np.vstack([self.a, new_a])
        self.b = np.vstack([self.b, new_b])

        self.params = self.solve()

