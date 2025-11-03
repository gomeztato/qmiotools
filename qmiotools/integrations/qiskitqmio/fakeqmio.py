from __future__ import annotations

from ...version import VERSION


import logging

logger=logging.getLogger('FakeQmio/'+VERSION)


from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from .qmiobackend import QmioBackend

def FakeQmio(calibration_file: str=None, thermal_relaxation: bool = True, temperature: float = 0 , gate_error: bool=False, readout_error: bool=False, logging_level: int=logging.NOTSET, logging_filename: str=None, qmio_partition: str="full", **kwargs) -> AerSimulator:
    r"""
    
    Create a Fake backend for Qmio that uses the last calibrations and AerSimulator. 

    Args:
        calibration_file (str): A path to a valid calibration file. If the path is **None** (default), the last calibration is loaded.
        thermal_relaxation (bool): If True, the noise model will include the thermal relaxation using the data from the calibration. Default: *True*
        temperature (float): the temperature in mK. If it is different of 0 (default). This is equivalent temperature used to calculate the probability of being |1\> due to thermal effects. See publication `arxiv:1412.2772 <https://arxiv.org/abs/1412.2772>`_. This temperature is passed to AerSimulator, that initially set it equally for all qubits, despite that could be different.
        gate_error (bool): Flag to include (True) or not (False. Default option) the gate errors in the model.
        readout_error (bool): Flag to include (True) or not (False. Default) the readout error from the calibrations file.
        logging_level (int): flag to indicate the logging level. Better if use the logging package levels. Default logging.NOTSET
        logging_filename (str):  Path to store the logging messages. Default *None*, i.e., output in stdout
        qmio_partition (str): qmio partition name on which you want to execute the code. Default 'full' which represents the entire decive. The rest of partitions are called 'isle1' 'isle2' and 'isle3'.
        **kwargs: other parameters to pass directly to :class:`qiskit_aer.AerSimulator`

    Returns:
        AerSimulator: A valid backend including the defined noise model.

    Raises:
        QmioException: if the configuration file could not be found. 

    """
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')    
    if logging_filename!=None:
        handler = logging.FileHandler(logging_filename)
    else:
        import sys
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(handler)

    logger.info("Logging FakeQmio started:")
    handler.flush()
    logger.info("Reading QmioBackend data")
    qmio=QmioBackend(calibration_file,logging_level=logging_level, logging_filename=logging_filename, qmio_partition=qmio_partition)
    noise_model = NoiseModel.from_backend(
        qmio, thermal_relaxation=thermal_relaxation,
        temperature=temperature,
        gate_error=gate_error,
        readout_error=readout_error)
    
    
    cls= AerSimulator.from_backend(qmio, noise_model=noise_model, **kwargs)
    
    cls.name = "FakeQmio"
    cls.description ="Fake backend for Qmio that uses the last calibrations and AerSimulator"
    #cls.version=VERSION
    logger.info("Created AerSimulator for Qmio with calibration_file:%s, thermal_relaxation: %s, temperature: %.2fmK , gate_error:%s, readout_error: %s "%(qmio._calibration_file, thermal_relaxation, temperature, gate_error, readout_error))
    return cls
        
    
