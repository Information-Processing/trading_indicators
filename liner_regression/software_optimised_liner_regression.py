import numpy as np


class OptimisedSoftwareLR:
    def __init__(self, initial_data, collumn_headers): 
        #initial data is sxp matrix
        self.collumn_headers = collumn_headers

        self.num_params = len(initial_data[0]) #actually equal to num_params + 1 as last collumn is output
        self.num_samples = len(initial_data)

        self.a = np.array([np.append(row[:-1], 1) for row in initial_data])
        self.at = self.a.transpose()

        self.b = initial_data[:, -1].reshape(self.num_samples,1)

        self.ata = self.at @ self.a
        self.ata_inv = np.linalg.inv(self.ata)

        self.atb = self.at @ self.b

        self.params = self.ata_inv @ self.atb

    def print_equation(self):
        params = [param.item() for param in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="")
        for param_idx in range(1, self.num_params):
            print(f" + {params[param_idx]:.2f} * {self.collumn_headers[param_idx]}", end="") 

        print("\n")

    def stream_line(self, line):
        line_output = line[-1]
        q_vec = np.append(line[:-1], 1).reshape(1, self.num_params)
        qt_vec = q_vec.reshape(self.num_params, 1)

        ata_diff = qt_vec @ q_vec
        atb_diff = line_output * qt_vec

        self.ata += ata_diff
        self.atb += atb_diff

    def stream_chunk(self, lines):
        for line in lines:
            self.stream_line(np.array(line))
        self.recalculate_params()

    def stream_chunk_optimised(self, lines):
        line_output = lines[:, -1]
        q_mat = np.concatenate([lines[:,:-1], np.ones((lines.shape[0], 1))], axis=1)
        qt_mat = q_mat.transpose()

        ata_diff = qt_mat @ q_mat
        atb_diff = qt_mat @ line_output.reshape(-1, 1)

        self.ata += ata_diff
        self.atb += atb_diff

        self.recalculate_params()

    def recalculate_params(self):
        self.ata_inv = np.linalg.inv(self.ata)
        self.params = self.ata_inv @ self.atb

