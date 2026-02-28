import numpy as np

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]


class LinearRegression:
    def __init__(self, initial_data, collumn_headers): 
        #initial data is sxp matrix
        self.collumn_headers = collumn_headers

        self.num_params = len(initial_data[0]) #actually equal to num_params + 1 as last collumn is output
        self.num_samples = len(initial_data[0])

        self.a = np.array([np.append(row[:-1], 1) for row in initial_data])
        print(f"A:\n {self.a}")

        self.at = self.a.transpose()
        print(f"AT:\n {self.at}")

        self.b = initial_data[:, -1].reshape(self.num_samples,1)
        print(f"B:\n {self.b}")

        self.ata = self.at @ self.a
        print(f"ata:\n {self.ata}")

        self.ata_inv = np.linalg.inv(self.ata)
        print(f"ata inverse:\n {self.ata}")

        self.atb = self.at @ self.b
        print(f"atb:\n {self.atb}")

        self.params = self.ata_inv @ self.atb
        print(f"params:\n {self.params}")
        self.print_equation()

    def print_equation(self):
        params = [float(param) for param in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="") 
        for param_idx in range(1, self.num_params):
            print(f" + {params[param_idx]:.2f} * {self.collumn_headers[param_idx]}", end="") 

        print("\n")

    def stream_line(self, line):
        if len(line) != self.num_params

        


if __name__ == "__main__":
    collumn_headers = ['MA', '1'] #example_for_now
    input_data = np.array([[1,1],[2,3]])

    lr = LinearRegression(input_data, collumn_headers)

