import matplotlib.pyplot as plt
import numpy as np
import pickle
import shapely


def load_file_names(path):
    return np.loadtxt(f"{path}/data.csv", dtype=str)


class SwissData(list):
    def __init__(self, path):
        self.path = path
        self.file_names = np.loadtxt(f"{path}/data.csv", dtype=str)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with open(f"{self.path}/{self.file_names[idx]}.pkl", "rb") as f:
            return pickle.load(f)


def get_boundary(polygon):
    polygon = np.array(polygon)
    polygon = polygon[:, :2]
    polygon = shapely.Polygon(polygon)
    return np.array(polygon.exterior.xy)
