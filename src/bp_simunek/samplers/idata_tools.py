import os
import pathlib
import pickle
from arviz import InferenceData

def base_path() -> None:
    return pathlib.Path(__file__).parent.resolve()

def idata_path() -> None:
    return os.path.join(base_path(), "data", "idata")

def save_idata_to_file(
        idata: InferenceData,
        filename: str,
        folder_path: str = idata_path()) -> None:
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)

    if os.path.exists(path=path):
        with open(path, "wb") as file:
            pickle.dump(obj=idata, file=file)
    else:
        with open(path, "ab") as file:
            pickle.dump(obj=idata, file=file)

def read_idata_from_file(
        filename: str, 
        folder_path: str = idata_path()) -> InferenceData:
    path = os.path.join(folder_path, filename)
    try:
        with open(path, "rb") as file:
            idata = pickle.load(file=file)
    except:
        print("Error reading idata file")

    return idata
