import shapely

from .tria_complex import triangulate_polygon_with_holes
from .wall_utils.wall_utils import get_wall_triangles
from .swiss_fp_data.fp_utils import Mapping
from .swiss_fp_data.floorplan_data import load_floorplan
from beartype import beartype
from einops import rearrange, repeat
from functools import partial
import pickle
import numpy as np
import os
import re
import shutil

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType

from tripy import earclip

import logging
import multiprocessing as mp
from tqdm import tqdm

logger = logging.getLogger(__name__)

PAD_VALUE = -1  # always use -1 for padding values


"""
ORDERS will be used to order traingles in a sample.
0: User provides EXTENDED_BOUNDARY, ENTRANCE, and INTERNAL_WALLS (walls that cannot have windows).
"""
ORDERS = {
    0: (
        "EXTENDED_BOUNDARY",
        "ENTRANCE",
        "INTERNAL",
        "NEGATIVE_SPACE",
        "^DOOR---WINDOW---PERIMETER",
        "DOOR",
        "WINDOW",
        "PERIMETER",
    )
}


def order_faces(face_data, order, randomize=True):
    """
    Orders faces based on the order provided.
    """
    faces, names = face_data["faces"], face_data["names"]
    idx = []
    for x in order:
        if x.startswith("^"):
            unmatching_indices = [y for y in x[1:].split("---")]
            matching_indices = [
                Mapping.NameToIndex[y]
                for y in Mapping.NameToIndex
                if y not in unmatching_indices
            ]
        else:
            matching_indices = [Mapping.NameToIndex[x]]
        all_matching = [
            i for i, n in enumerate(names) if i not in idx and n in matching_indices
        ]
        if randomize:
            np.random.shuffle(all_matching)
        idx.extend(all_matching)
    faces = faces[idx]
    names = names[idx]
    face_data.update({"faces": faces, "names": names})
    return face_data


def get_edge_of_a_face(face):
    return [(a, b) for a, b in zip(np.asarray(face, dtype=float), np.roll(face, -1))]


def order_by_edges(
    face_data,
):
    """
    Orders faces based on the face_edges.
    """
    faces, names = face_data["faces"], face_data["names"]
    unique_names = []
    all_ordered_idx = []
    for n in names:
        if n not in unique_names:
            unique_names.append(n)
        else:
            continue
        idx = torch.where(names == n)[0]

        ordered_idx = [idx[0]]
        open_edges = get_edge_of_a_face(faces[ordered_idx[-1]])
        while len(ordered_idx) < len(idx):
            found = False
            for i in idx:
                if i in ordered_idx:
                    continue
                edges = get_edge_of_a_face(faces[i])
                for ie, oe in enumerate(open_edges):
                    if (oe in edges) or (tuple(reversed(oe)) in edges):
                        ordered_idx.append(i)
                        open_edges.pop(ie)
                        for e in get_edge_of_a_face(faces[i]):
                            if not ((e == oe) or (tuple(reversed(e))) == oe):
                                open_edges.append(e)
                        found = True
                        break
                if found:
                    break
            if not found and len(ordered_idx) < len(idx):
                ordered_idx.append(next(i for i in idx if i not in ordered_idx))
                open_edges = get_edge_of_a_face(faces[ordered_idx[-1]])
        all_ordered_idx.extend(ordered_idx)
    # import pdb; pdb.set_trace()
    all_ordered_idx = torch.LongTensor(all_ordered_idx)
    face_data.update({"faces": faces[all_ordered_idx], "names": names[all_ordered_idx]})
    return face_data


def create_walls(spaces, perimeter):
    walls = shapely.Polygon(perimeter[:, :2])
    for s in spaces:
        space = shapely.Polygon(spaces[s][:, :2])
        walls = shapely.difference(walls, space)

    walls = walls.buffer(0)
    assert walls.is_valid and walls.is_simple and walls.geom_type == "Polygon", walls is walls.geom_type

    triangles = triangulate_polygon_with_holes(walls)
    return np.array(triangles)#, np.ones(len(triangles)) * Mapping.NameToIndex["PERIMETER"]

    # all_ts = []
    # for t in get_wall_triangles(dict(perimeter=perimeter, spaces=spaces)):
    #     all_ts.append(np.array(t.exterior.coords)[:-1])
    # all_ts = np.array(all_ts)
    # return all_ts


def check_sample(i, path, max_size, remove_with_errors):
    f = load_floorplan(f"{path}/{i}.pkl")
    n = f.perimeter.reshape(-1).max()
    if n > max_size:
        print(f"Removing {i} as size is {n}")
        return i
    for e in f.errors:
        for r in remove_with_errors:
            if re.fullmatch(r, e):
                print(i, e)
                return i
    return None


def check(i, d):
    try:
        d.__getitem__(i)
        return i
    except:
        print(f"Invalid sample: {i}")


def process_samples(path, nums, keys):
    d = SwissDataset(path, nums, keys=keys, reload=True)
    p_check = partial(check, d=d)
    with mp.Pool(mp.cpu_count()) as pool:
        processed_samples = list(tqdm(pool.imap(p_check, d.data), total=len(d.data)))
    processed_samples = [x for x in processed_samples if x is not None]
    np.savetxt(f"{d.path}/data-2.csv", processed_samples, fmt="%s", delimiter="\t")


class SwissDataset(Dataset):
    """
    A dataset class to load floorplan data from a folder.
    :param str path: Path of the dataset folder containing .csv and .pkl files.
    :param int nums: Number of samples to load. If None, load all samples.
    :param bool reload: Whether to reload and preprocess the data.
    """

    def __init__(self, path, nums, start_idx=0, reload=True, keys=None):
        self.path = path
        self.data = np.loadtxt(f"{self.path}/data.csv", dtype="str", delimiter="\t")
        self.data = sorted(self.data, key=lambda x: x[1])
        self.original_size = len(self.data)
        print(
            f"Loading samples [{start_idx}-{start_idx + nums}] / [{self.original_size}]"
        )

        total_samples = len(self.data)
        if start_idx >= total_samples:
            raise ValueError(
                f"start_idx {start_idx} exceeds dataset size {total_samples}"
            )
        if start_idx + nums > total_samples:
            logger.warning(
                f"Total samples: {total_samples}, Start from: {start_idx}, Nums: {nums}"
            )
            nums = total_samples - start_idx
            assert nums > 0, f"Total samples: {total_samples}, Start from: {start_idx}"
        self.data = self.data[start_idx : start_idx + nums]
        logger.info(f"Using samples from index {start_idx} to {start_idx + nums}")

        self.processed_dir = f"{self.path}/processed"
        if reload:
            shutil.rmtree(self.processed_dir, ignore_errors=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.max_num_faces = 128
        self.max_size = 25600
        self.factor = 0.001
        self.collate_fn = partial(collate_fn, pad_value=PAD_VALUE)
        self.keys = keys
        logger.info(f"Samples Loaded - {len(self.data)} out of {self.original_size}")

    def remove_errored_samples(self):
        remove_with_errors = [
            r"No polyline in PERIMETER layer.",
            r"More than one polyline in PERIMETER layer.",
            # r"No polyline in BO-SERVICE layer.",
            # r"No polyline in BO-INTERNAL layer.",
            r"Invalid space .+.",
            # r"Invalid door/window .+.",
            r"Invalid layer .+.",
            # r"No NORTH found.",
            r"Overlapping objects .+ and .+.",
            # r"Object .+ is outside the perimeter.",
            # r"No access for .+.",
        ]
        to_remove = []
        with mp.Pool(mp.cpu_count()) as pool:
            to_remove = pool.starmap(
                check_sample,
                [(i, self.path, self.max_size, remove_with_errors) for i in self.data],
            )
            to_remove = [i for i in to_remove if i is not None]
        print(f"Removing {len(to_remove)} out of {len(self.data)} samples.")
        self.data = [x for x in self.data if x not in to_remove]
        np.savetxt(f"{self.path}/data.csv", self.data, fmt="%s", delimiter="\t")

    def scale(self, sample):
        sample.perimeter = np.array(sample.perimeter) * self.factor
        sample.spaces = {x: np.array(y) * self.factor for x, y in sample.spaces.items()}
        sample.door_windows = {
            x: np.array(y) * self.factor for x, y in sample.door_windows.items()
        }
        sample.boundaries = {
            x: np.array(y) * self.factor for x, y in sample.boundaries.items()
        }

    def create_negative_space(self, sample):
        x = shapely.difference(
            shapely.Polygon(sample.boundaries["EXTENDED_BOUNDARY-1"]),
            shapely.Polygon(sample.perimeter),
        )
        if x.geom_type == "Polygon":
            if x.area > 2:
                sample.boundaries["NEGATIVE_SPACE-1"] = np.array(x.exterior.coords)
        elif x.geom_type == "MultiPolygon":
            for i, p in enumerate(x.geoms):
                if p.area > 2:
                    sample.boundaries[f"NEGATIVE_SPACE-{i}"] = np.array(
                        p.exterior.coords
                    )

    def triangulate(self, sample):
        for key in sample.spaces:
            sample.spaces[key] = triangulate(sample.spaces[key])
        for key in sample.door_windows:
            sample.door_windows[key] = triangulate(sample.door_windows[key])
        for key in sample.boundaries:
            sample.boundaries[key] = triangulate(sample.boundaries[key])

    @classmethod
    def NameToIndex(cls, n):
        base_name = Mapping.NameToIndex.get(n.split("-")[0])
        # try:
        #     index = int(n.split("-")[1]) - 1
        # except IndexError:
        #     index = 0
        return base_name  # + index

    def generate_caption(self, sample):
        total_area = shapely.Polygon(sample.perimeter).area
        num_rooms = len(
            [x for x in sample.spaces if "ROOM" in x or "LIVING" in x or "OFFICE" in x]
        )
        desc = "It is %i room apartment with a total area of %d sq. m. " % (
            num_rooms,
            total_area,
        )
        for i in sample.spaces:
            desc += "%s has an area of %2d sq. m. " % (
                i.split("-")[0],
                shapely.Polygon(sample.spaces[i]).area,
            )
        for i in sample.door_connections:
            desc += "%s is connected to %s. " % (
                sample.door_connections[i][0].split("-")[0],
                sample.door_connections[i][1].split("-")[0],
            )
        return desc

    def get_vertices_faces(self, sample):
        all_faces = np.concatenate(
            [
                sample.perimeter,
                np.concatenate(list(sample.spaces.values())),
                np.concatenate(list(sample.door_windows.values())),
                np.concatenate(list(sample.boundaries.values())),
            ]
        )
        names = [
            self.NameToIndex("PERIMETER"),
        ] * len(sample.perimeter)
        for i in sample.spaces:
            names.extend(
                [
                    self.NameToIndex(i),
                ]
                * len(sample.spaces[i])
            )
        for i in sample.door_windows:
            names.extend(
                [
                    self.NameToIndex(i),
                ]
                * len(sample.door_windows[i])
            )
        for i in sample.boundaries:
            names.extend(
                [
                    self.NameToIndex(i),
                ]
                * len(sample.boundaries[i])
            )

        flat_vertices = all_faces.reshape(-1, 2)
        unique_vertices, indices = np.unique(flat_vertices, axis=0, return_inverse=True)
        faces = np.array(
            [
                [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]
                for i in range(all_faces.shape[0])
            ]
        )
        sorted_pairs = sorted(zip(names, faces), key=lambda x: x[0])
        sorted_names, sorted_faces = zip(*sorted_pairs)

        sorted_names = np.array(sorted_names)
        sorted_faces = np.array(sorted_faces)
        return dict(
            vertices=torch.Tensor(unique_vertices),
            faces=torch.LongTensor(sorted_faces),
            names=torch.LongTensor(sorted_names).unsqueeze(-1),
        )

    def transform(self, sample):
        data = {}
        sample.reposition(sample.get_origin())
        self.scale(sample)
        data["texts"] = self.generate_caption(sample)
        
        self.create_negative_space(sample)
        sample.perimeter = create_walls(sample.spaces, sample.perimeter)

        self.triangulate(sample)
        face_data = self.get_vertices_faces(sample)
        face_data = order_faces(face_data, order=ORDERS[0])
        face_data = order_by_edges(face_data)
        data.update(face_data)

        data["face_edges_2"] = get_edges(face_data["faces"])
        data["face_edges_1"] = get_edges(face_data["faces"], allow_point=True)
        return data

    def preprocess(self, i):
        processed_file = f"{self.processed_dir}/{self.data[i]}.pkl"
        try:
            with open(processed_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            data = load_floorplan(f"{self.path}/{self.data[i]}.pkl")
            data = self.transform(data)
            data["path"] = processed_file
            data["num_faces"] = len(data["faces"])
            with open(processed_file, "wb") as f:
                pickle.dump(data, f)
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = self.data.index(idx)
        data = self.preprocess(idx)
        if self.keys is not None:
            return {k: data[k] for k in self.keys}
        return data


def triangulate(points):
    """
    Triangulates a polygon.
    """
    if all(points[0] == points[-1]):
        points = points[:-1]
    if points.shape[1] == 3:
        points = points[:, :2]
    points = shapely.Polygon(points)
    points = points.buffer(0)
    triangles = earclip(points.exterior.coords[:-1])
    triangles = np.array(triangles)
    return triangles


def get_vertices_faces(perimeter):
    all_faces = perimeter
    names = [
        0,
    ] * len(perimeter)
    flat_vertices = all_faces.reshape(-1, 2)
    unique_vertices, indices = np.unique(flat_vertices, axis=0, return_inverse=True)
    faces = np.array(
        [
            [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]
            for i in range(all_faces.shape[0])
        ]
    )
    sorted_pairs = sorted(zip(names, faces), key=lambda x: x[0])
    sorted_names, sorted_faces = zip(*sorted_pairs)
    sorted_names = np.array(sorted_names)
    sorted_faces = np.array(sorted_faces)

    vertices = torch.Tensor(unique_vertices)
    faces = torch.LongTensor(sorted_faces)
    names = torch.LongTensor(sorted_names).unsqueeze(-1)
    face_edges = get_edges(faces)

    return dict(
        vertices=vertices,
        faces=faces,
        names=names,
        face_edges=face_edges,
    )


def collate_fn(batch, pad_value):
    """
    A collate function to create equal size samples using a pad_value.
    """
    is_dict = isinstance(batch[0], dict)
    if is_dict:
        keys = batch[0].keys()
        batch = [d.values() for d in batch]
    output = []
    for datum in zip(*batch):
        if torch.is_tensor(datum[0]):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_value)
        else:
            datum = list(datum)
        output.append(datum)
    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))
    return output


def remove_padding(sequence, padding_value=PAD_VALUE):
    """
    Removes the padding from a padded dataset.
    """
    mask = torch.any(sequence != padding_value, dim=-1)
    cleaned_sequence = sequence[mask]
    return cleaned_sequence


@beartype
def get_edges(faces: TensorType["num_faces", 3, 2, int], allow_point=False):  # noqa: F821
    """
    Return edges of faces sharing exaclty two vertices.
    :param Tensor faces: tensor of faces of a shape ('num_faces', 3)
    """
    num_faces = len(faces)
    all_edges = torch.stack(
        torch.meshgrid(
            torch.arange(num_faces, device=faces.device),
            torch.arange(num_faces, device=faces.device),
            indexing="ij",
        ),
        dim=-1,
    )

    first_vertices = rearrange(faces, "i c -> i 1 c 1")
    second_vertices = rearrange(faces, "j c -> 1 j 1 c")
    shared_vertices = first_vertices == second_vertices
    num_shared_vertices = shared_vertices.any(dim=-1).sum(dim=-1)
    if allow_point:
        is_neighbor_face = num_shared_vertices == 1
    else:
        is_neighbor_face = num_shared_vertices == 2
    face_edge = all_edges[is_neighbor_face]
    return torch.as_tensor(face_edge, dtype=torch.long, device=faces.device)


def get_perimeter(
    perimeter=[[0.0, 0.0], [10000.0, 0.0], [10000.0, 10000.0], [0, 10000.0]],
    batch_size=1,
):
    perimeter = triangulate(np.array(perimeter))
    names = [
        SwissDataset.NameToIndex("PERIMETER"),
    ] * len(perimeter)

    flat_vertices = perimeter.reshape(-1, 2)
    unique_vertices, indices = np.unique(flat_vertices, axis=0, return_inverse=True)
    faces = np.array(
        [
            [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]
            for i in range(perimeter.shape[0])
        ]
    )
    face_edges = get_edges(torch.LongTensor(faces))
    return dict(
        vertices=repeat(torch.Tensor(unique_vertices), "... -> b ...", b=batch_size),
        faces=repeat(torch.LongTensor(faces), "... -> b ...", b=batch_size),
        names=repeat(
            torch.LongTensor(names).unsqueeze(-1), "... -> b ...", b=batch_size
        ),
        face_edges=repeat(face_edges, "... -> b ...", b=batch_size),
    )
