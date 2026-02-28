import numpy as np

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]

LOG_LVL = 1

def log(message, level):
    if (level <= LOG_LVL):
        print(message)


class LinearRegression:
    def __init__(self, initial_data, collumn_headers): 
        #initial data is sxp matrix
        self.collumn_headers = collumn_headers

        self.num_params = len(initial_data[0]) #actually equal to num_params + 1 as last collumn is output
        self.num_samples = len(initial_data[0])

        self.a = np.array([np.append(row[:-1], 1) for row in initial_data])
        log(f"A:\n {self.a}", 2)

        self.at = self.a.transpose()
        log(f"AT:\n {self.at}", 2)

        self.b = initial_data[:, -1].reshape(self.num_samples,1)
        log(f"B:\n {self.b}", 2)

        self.ata = self.at @ self.a
        log(f"ata:\n {self.ata}", 2)

        self.ata_inv = np.linalg.inv(self.ata)
        log(f"ata inverse:\n {self.ata}", 2)

        self.atb = self.at @ self.b
        log(f"atb:\n {self.atb}", 2)

        self.params = self.ata_inv @ self.atb
        log(f"initial params:\n {self.params}", 1)
        self.print_equation()

    def print_equation(self):
        params = [param.item() for param in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="")
        for param_idx in range(1, self.num_params):
            print(f" + {params[param_idx]:.2f} * {self.collumn_headers[param_idx]}", end="") 

        print("\n")

    def stream_line(self, line):
        if len(line) != self.num_params:
            log("ERROR: line and param size diff", 0)
            return
        
        line_output = line[-1]
        q_vec = np.append(line[:-1], 1).reshape(1, self.num_params)
        qt_vec = q_vec.reshape(self.num_params, 1)

        log(f"q:\n {q_vec}", 2)
        log(f"qt:\n {qt_vec}", 2)

        ata_diff = qt_vec @ q_vec
        atb_diff = line_output * qt_vec

        log(f"ata diff:\n {ata_diff}", 2)
        log(f"atb diff:\n {atb_diff}", 2)

        self.ata += ata_diff
        self.atb += atb_diff

    def recalculate_params(self):
        self.ata_inv = np.linalg.inv(self.ata)
        self.params = self.ata_inv @ self.atb
        print("recalculated params:")

        log(f"\n {self.params}", 1)





if __name__ == "__main__":
    collumn_headers = ['MA', '1'] #example_for_now
    input_data = np.array([[1,1],[2,3]])

    lr = LinearRegression(input_data, collumn_headers)
    append_line = np.array([4,5])
    lr.stream_line(append_line)
    lr.recalculate_params()

