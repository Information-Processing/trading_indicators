import numpy as np

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]
class LinearRegression:
    def __init__(self, initial_data): 
        #initial data is sxp matrix
        self.num_params = len(initial_data[0])
        self.num_samples = len(initial_data[0])

        self.cur_a = np.array([np.append(row[:-1], 1) for row in initial_data])
        print(f"A:\n {self.cur_a}")

        self.cur_at = self.cur_a.transpose()
        print(f"AT:\n {self.cur_at}")

        self.cur_b = initial_data[:, -1].reshape(self.num_samples,1)
        print(f"B:\n {self.cur_b}")

        self.cur_ata = self.cur_at @ self.cur_a
        print(f"ata:\n {self.cur_ata}")

        self.cur_ata_inv = np.linalg.inv(self.cur_ata)
        print(f"ata inverse:\n {self.cur_ata}")

        self.cur_atb = self.cur_at @ self.cur_b
        print(f"atb:\n {self.cur_atb}")

        self.cur_params = self.cur_ata_inv @ self.cur_atb
        print(f"params:\n {self.cur_params}")


if __name__ == "__main__":
    input_data = np.array([[1,1],[2,3]])
    lr = LinearRegression(input_data)

