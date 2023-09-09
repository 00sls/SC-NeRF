from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .spaces import DsetSpaces
from .shiny import MVSDatasetShiny
from .real import RealEstateDataset
dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'shiny': MVSDatasetShiny,
                'real': RealEstateDataset,
                'spaces': DsetSpaces}