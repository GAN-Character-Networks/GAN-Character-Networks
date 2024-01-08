r""" This module contains the output logger classes which are used to save the output data of our system.

Authors
--------
 * Adel Moumen 2023
"""
import os
from typing import Dict, List
import logging
import json

logger = logging.getLogger(__name__)


class OutputLogger:
    """A class for managing the output loggers which are saving the output data as a unified format
    Additionnal loggers can be added by simply sub-classing this class.

    Args:
        save_path : str
            The path where the output predictions will be saved
    """

    def __init__(self, save_path: str) -> None:

        self.save_path = save_path
        self._check_path()

    def _check_path(self) -> None:
        """Check if the save path exists and create it if not"""
        directory = os.path.dirname(self.save_path)
        os.makedirs(directory, exist_ok=True)

    def _check_file(self) -> None:
        """Check if the save file exists and create it if not"""
        if not os.path.isfile(self.save_path):
            open(self.save_path, "w").close()

    def __call__(self, data: List[Dict]) -> None:
        """Save the input list of dictionnary as a log file"""
        raise NotImplementedError


class JSONLogger(OutputLogger):
    """A output logger which save the predictions as a unified JSON format

    Args:
        save_path : str
            The path where the output predictions will be saved
    """

    def __init__(self, save_path: str) -> None:
        super().__init__(save_path=save_path)

    def __call__(self, data: List[Dict]) -> None:
        """Save the input list of dictionnary as a JsonL file

        Args:
            data : List[Dict]
                The data structure to save to the JSON format.
        """
        self._check_file()
        with open(self.save_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        logger.info(f"Saved data to {self.save_path}")
