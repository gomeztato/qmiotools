from qiskit.providers import QubitProperties, BackendV2, Options, Job
from qiskit.providers import JobStatus, JobV1  

#Removed for integration with Qiskint 2.0
#from qiskit.providers.models.backendstatus import BackendStatus

from qiskit.circuit.gate import Instruction
from qiskit.circuit.library import ECRGate, IGate, Measure, RZXGate, RZGate, SXGate,ECRGate, XGate
from qiskit.circuit import Delay
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import UGate, CXGate, Measure
from qiskit.circuit import Parameter, QuantumCircuit, ClassicalRegister
from qiskit.converters import circuit_to_dag
try:
    from qiskit.pulse import Schedule, ScheduleBlock
    from .opexporter import OPExporter
except:
    import warnings
    warnings.warn("Using a Qiskit version that does not support pulses. Pulses will not be available")
    class Schedule():
        {}
    class ScheduleBlock():
        {}
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.result import Result, Counts  
#Removed for integration with Qiskint 2.0
#from qiskit.qobj import QobjExperimentHeader
from qiskit import qasm2, qasm3, transpile
#Removed for integration with Qiskint 2.0
#from qiskit.qobj.utils import MeasLevel

import numpy as np
import math
import uuid
import atexit
#import datetime
from collections import Counter, OrderedDict
from datetime import date,datetime
from typing import Union, List, Optional, Dict,  Any, TYPE_CHECKING, Union, cast

import warnings
import re

from ...exceptions import QPUException, QmioException
from ..utils import Calibrations 
from ...version import VERSION
from ...data import QBIT_MAP, QUBIT_POSITIONS
from .qmiojob import QmioJob
from .flattencircuit import FlattenCircuit



from .qasmcircuit import QasmCircuit


from qmio import QmioRuntimeService
from qmio.backends import QPUBackend


import logging


DEFAULT_OPTIONS=Options(shots=10000,memory=False,repetition_period=None,res_format="binary_count",output_qasm3=False)
FORMATS=["binary_count","raw","binary","squash_binary_result_arrays"]
DT=0.5*1e-9 #0.5ns

class QmioBackend(BackendV2):
    """
    Backend to execute Jobs in Qmio QPU.
    
    Args:
            calibration_file (Str or None):  full path to a calibration file. Default *None* and loads the last calibration file from the directory indicated by environment QMIO_CALIBRATIONS
            
            logging_level (int): flag to indicate the logging level. Better if use the :py:mod:`logging` package levels. Default :py:data:`logging.NOTSET`
            
            logging_filename (str):  Path to store the logging messages. Default *None*, i.e., output in stdout
            
            tunnel_time_limit (str): time limit user specified for stablish interactive tunnels (for example, "00:15:00" indicates a limit of 15 minutes)
            
            reservation_name (str): reservation name user specified

            qmio_partition (str): qmio partition name on which you want to execute the code. Default 'full' which represents the entire decive. The rest of partitions are called 'isle1' 'isle2' and 'isle3'.
            
            kwargs: Other parameters to pass to Qiskit :py:class:`qiskit.providers.BackendV2` class
            
            

    
    It uses :py:class:`qmio.QmioRuntimeService` to submit circuits to the QPU. By default, the calibrations are read from the last JSON file in the directory set by environ variable QMIO_CALIBRATIONS, but accepts a direct filename to use instead of.
    
    .. attention:: 
        The execution using the method run is synchronous.
    
    To create a new backend use::
    
        backend=QmioBackend()
    
    It will print the file where the calibration were read in. If you want to use a specific calibrations file, use::
        
        backend=QmioBackend(<file name>)

    where <file name> is the path to the calibrations file that must be used. If the file is not found, it raises a exception.

    **Example**::

        from qiskit.circuit import QuantumCircuit
        from qiskit import transpile
        from qmiotools.integrations.qiskitqmio import QmioBackend
       
        backend=QmioBackend() # loads the last calibration file from the directory $QMIO_CALIBRARTIONS
        
        # Creates a circuit qwith 2 qubits 
        c=QuantumCircuit(2)
        c.h(0)
        c.h(0)
        c.measure_all()

        #Transpile the circuit using the optimization_level equal to 2
        c=transpile(c,backend,optimization_level=2)
   
        #Execute the circuit with 1000 shots. Must be executed from a node with a QPU.
        job=backend.run(c,shots=1000)

        #Return the results
        print(job.result().get_counts())
    
   
    """
    

    def __init__(self, calibration_file: str=None, logging_level: int=logging.NOTSET, logging_filename: str=None,
                 tunnel_time_limit: str=None,
                 reservation_name: str=None,
                 qmio_partition: str="full",**kwargs):
        
        self._provider=None
        self._name="Qmio"
        self._description="CESGA Qmio"
        self._backend_version=VERSION
        self._logger = logging.getLogger("QmioBackend/%s"%VERSION)
        self._QPUBackend=None
        self._calibration_file=None
        self._exporter=None
        self._reservation_name=reservation_name
        self._tunnel_time_limit=tunnel_time_limit
        self._qmio_partition=qmio_partition


        #
        # Logging activate
        #

        self._logger.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')    
        if logging_filename!=None:
            self._handler = logging.FileHandler(logging_filename)
        else:
            import sys
            self._handler = logging.StreamHandler(sys.stdout)
        
        self._handler.setFormatter(formatter)
        if (self._logger.hasHandlers()):
            self._logger.handlers.clear()
        self._logger.addHandler(self._handler)
        
        self._logger.info("Logging started:")
        self._handler.flush()
        
        #
        # Activate Exit
        #
        
        atexit.register(self.__exit__)
        
        #
        # Super init
        #
            
        super().__init__(self,name=self._name, description=self._description, **kwargs)
        
        self._logger.info("Loading calibrations")
        self._handler.flush()
        
        calibrations=Calibrations.import_last_calibration(calibration_file)
        self._calibration_file=calibrations.get_filename()
        properties=[]
        qubits=calibrations.get_qubits()
        qubit_conections=calibrations.get_mapping()
        
        
        ########

        isle1_qubits=[13,19,20,21,22,23,29,30,31]
        isle1_conections=[12,19,20,21,22,23,31,32]

        isle2_qubits=[6,10,11,12,14]
        isle2_conections=[6,10,11,13]

        isle3_qubits=[8,15,16,25,26]
        isle3_conections=[15,16,26,27]
        
        ########
        
        if self._qmio_partition=="full":
            log_to_phys=QBIT_MAP.copy()
            isle=[i for i in range(len(log_to_phys))]
            qubit_keys=list(qubits.keys())
            keys_pairs=qubit_conections

        elif self._qmio_partition=="isle1":
            isle=isle1_qubits
            log_to_phys=[]
            qubit_keys=[]
            for i in isle1_qubits:
                log_to_phys.append(QBIT_MAP[i])
                qubit_keys.append(list(qubits.keys())[i])
            keys_pairs=[]
            for i in isle1_conections:
                keys_pairs.append(qubit_conections[i])
            

        elif self._qmio_partition=="isle2":
            isle=isle2_qubits
            log_to_phys=[]
            qubit_keys=[]
            for i in isle2_qubits:
                log_to_phys.append(QBIT_MAP[i])
                qubit_keys.append(list(qubits.keys())[i])
            keys_pairs=[]
            for i in isle2_conections:
                keys_pairs.append(qubit_conections[i])
        elif self._qmio_partition=="isle3":
            isle=isle3_qubits
            log_to_phys=[]
            qubit_keys=[]
            for i in isle3_qubits:
                log_to_phys.append(QBIT_MAP[i])
                qubit_keys.append(list(qubits.keys())[i])
            keys_pairs=[]
            for i in isle3_conections:
                keys_pairs.append(qubit_conections[i])
        else:
            raise QmioException("Specify a valid Qmio partition: full, isle1, isle2 or isle3")         
                
        self._log_to_phys_mapping=log_to_phys
        self._isle_to_log_mapping=qubit_keys

        num_qubits=len(log_to_phys)
        QBIT_LIST=[i for i in range(num_qubits)]
        
        #
        # Load Qubits Properties
        #

        j=0
        for i in range(max(QBIT_LIST)+1):
            if i in QBIT_LIST:
                key=qubit_keys[j]
                properties.append(QubitProperties(t1=qubits[key]["T1 (s)"],t2=qubits[key]["T2 (s)"],frequency=qubits[key]["Drive Frequency (Hz)"]))
                j=j+1
                self._logger.debug("Qubit:%s, T1=%.9f, T2=%.9f, Drive Freq:%f"%(str((i,)),qubits[key]["T1 (s)"],qubits[key]["T2 (s)"],qubits[key]["Drive Frequency (Hz)"]))
            else:
                properties.append(None)
                
        self._logger.info("Number of loaded qubits %d"%len(properties))
        
        
        target = Target(description="qmio", num_qubits=len(properties), dt=DT, granularity=1, 
                        min_length=1, pulse_alignment=1, acquire_alignment=1, 
                        qubit_properties=properties, concurrent_measurements=None)
        
        
        theta = Parameter('theta')
        
        errors=calibrations.get_1Q_errors()
        durations=calibrations.get_1Q_durations()

        sx_inst=OrderedDict()
        x_inst=OrderedDict()

        for i in range(num_qubits):
            sx_inst[(i,)]=InstructionProperties(duration=durations[(int(qubit_keys[i][2:-1]),)], error=errors[(int(qubit_keys[i][2:-1]),)])
            self._logger.debug("Added SX[%d]- Duration %.10fs - error %f"%(i,durations[(int(qubit_keys[i][2:-1]),)],errors[(int(qubit_keys[i][2:-1]),)]))
        for i in range(num_qubits):
            x_inst[(i,)]=InstructionProperties(duration=durations[(int(qubit_keys[i][2:-1]),)]*2, error=errors[(int(qubit_keys[i][2:-1]),)])
            self._logger.debug("Added X[%d]- Duration %.10fs - error %f"%(i,durations[(int(qubit_keys[i][2:-1]),)]*2,errors[(int(qubit_keys[i][2:-1]),)]))
        
        target.add_instruction(SXGate(), sx_inst)
        target.add_instruction(XGate(), x_inst)
        
        rz_inst=OrderedDict()
        for i in range(num_qubits):
            rz_inst[(i,)]=InstructionProperties(duration=0.0)
            self._logger.debug("Added rz[%d]- Duration %.10fs - error %f"%(i,0.0,0.0))
                
        target.add_instruction(RZGate(theta), rz_inst)
        
        #q2_inst=calibrations.get_2Q_errors()
        #target.add_instruction(ECRGate(), q2_inst)   

        
        errors=calibrations.get_2Q_errors()
        durations=calibrations.get_2Q_durations()
        
        #self._logger.debug(durations)
        
        ecr_inst=OrderedDict()
        for i in range(len(keys_pairs)):
            logical_pair=(isle.index(keys_pairs[i][0]),isle.index(keys_pairs[i][1])) 
            ecr_inst[logical_pair]=InstructionProperties(duration=durations[keys_pairs[i]], error=errors[keys_pairs[i]])
            self._logger.debug("Added ecr_inst[(%s) - duration %.10fs - error %f]"%(str(logical_pair),durations[keys_pairs[i]],errors[keys_pairs[i]]))
            
        
        target.add_instruction(ECRGate(), ecr_inst)
        #target.add_instruction(RZXGate(math.pi/4), ecr_inst, name="rzx(pi/4)")
        
        measures=OrderedDict()

        errors=calibrations.get_measuring_errors()
        durations=calibrations.get_measuring_durations()

        #for i in qubits:
        
        for i in range(num_qubits):
            measures[(i,)]=InstructionProperties(duration=durations[(int(qubit_keys[i][2:-1]),)], error=errors[(int(qubit_keys[i][2:-1]),)])
            self._logger.debug("measures[%d] - duration %.10fs - error %f"%(i,durations[(int(qubit_keys[i][2:-1]),)],errors[(int(qubit_keys[i][2:-1]),)]))
            
        target.add_instruction(Measure(),measures)
        
        delays=OrderedDict()
        for i in range(num_qubits):
            delays[(i,)]=None
        
        target.add_instruction(Delay(Parameter("t")),delays)
                               
        self._target = target
        
        self._options = DEFAULT_OPTIONS
        self._logger.info("Default options %s"%DEFAULT_OPTIONS)
        
        self._version = VERSION
        self._logger.info("VERSION %s"%VERSION)
        
        self._max_circuits=1000
        self._logger.info("MAX CIRCUITS %d"%self._max_circuits)
        
        self._max_shots=self._options.get("shots")*10
        self._logger.info("MAX SHOTS PER STEP %d"%self._max_shots)
        
        self.max_shots=self._max_circuits*self._max_shots
        self._logger.info("MAX SHOTS PER JOB %d"%self.max_shots)
        

    @property
    def target(self) -> Target:
        return self._target
    
    @classmethod
    def _default_options(cls):
        return DEFAULT_OPTIONS
    
    @classmethod
    def formats(cls):
        """
            Returns the possible output formats supported by QMIO.
        """
        return FORMATS
    
    @property
    def max_circuits(self):
        return self._max_circuits
    
    def connect(self):
        """
            This method connect to the QPU. You do not need to connect, but you can if you want. If a connection exits, it is closed and the class is reconnected again

        """
        self._logger.debug("Connecting backend")
        if self._QPUBackend is not None:
            self._QPUBackend.disconnect()
        else:
            self._logger.info("Connecting with parameters - reservation_name %s - tunnel_time_limit %s"%(self._tunnel_time_limit,self._reservation_name))
            self._QPUBackend=QPUBackend(tunnel_time_limit=self._tunnel_time_limit, reservation_name=self._reservation_name)
        
        self._QPUBackend.connect()
    
    def disconnect(self):
        """
            This method closes the connection with the QPU and destroy the QPUBackend instance. It is recommended to do it before exit the program
        """
        self._logger.debug("Disconnecting  backend")
        if self._QPUBackend is not None:
            self._QPUBackend.disconnect()
            del self._QPUBackend
            self._QPUBackend=None
            
    def __del__(self):
        """
            Internal method to call when the instance of this class is deleted.
        """
        
        if self._handler is not None:
            self._logger.debug("Deleting instance of QmioBackend")
            self._logger.removeHandler(self._handler)
            self._handler=None
        
        self._close()
        atexit.unregister(self.__exit__)
    
    def __exit__(self):
        """
            Internal method to call on exit. Do not call directly
        """
        if self._handler is not None:
            self._logger.debug("Deleting instance of QmioBackend")
            self._logger.removeHandler(self._handler)
            self._handler=None
        self._close()
        atexit.unregister(self.__exit__)
        
    def _close(self):
        self.disconnect()
        del self._QPUBackend
        self._QPUBackend=None
    
    def _to_qasm3(self,c):
        self._logger.debug("Converting to OPENQASM 3.0")
        basis_gates=self.operation_names.copy()
        basis_gates.remove('measure')
        basis_gates.remove('delay')
        qasm=qasm3.dumps(c, includes=[], basis_gates=basis_gates).replace("\n","")
        self._logger.debug("Obtainded QASM from circuit:%s"%qasm.replace("\n",""))
        if "qubit[" in qasm:
            c=transpile(c,self,optimization_level=0)
            qasm=qasm3.dumps(c, includes=[], basis_gates=basis_gates).replace("\n","")
        mapping=self._log_to_phys_mapping
        for i in range(self.num_qubits-1,-1,-1):
            qasm=qasm.replace("$%d;"%i,"$%d;"%mapping[i])
            qasm=qasm.replace("$%d,"%i,"$%d,"%mapping[i])
            qasm=qasm.replace("$%d "%i,"$%d "%mapping[i])
        
        #self._logger.info("Replacing SC gate by RX(pi/2) as a temporal fix")
        #qasm=qasm.replace("SX ","rx(pi/2) ").replace("sx ","rx(pi/2) ")
        #self._logger.debug("Final submitted circuit %s"%qasm)
        return qasm
            
    def _to_qasm2(self,c):

        all_qubits=c.qubits
        cdag=circuit_to_dag(c)
        active_qubits = [qubit for qubit in all_qubits if qubit not in cdag.idle_wires()]
        mapping=self._isle_to_log_mapping

        self._logger.debug("Converting to OPENQASM 2.0")
        qasm=qasm2.dumps(c)
        self._logger.debug("Circuit to transform:\n%s"%qasm )
        qasm=re.sub("\\ngate rzx.*\\n","\\n",qasm)
        #qasm=re.sub("\\nopaque delay.*","",qasm)
        qasm=re.sub("\\ngate ecr.*\\n","\\ngate ecr q0, q1 {};\\n",qasm)

        qasm = re.sub(r'\bq\[%s\]'%(str(len(all_qubits))), 'TEMP_Q%s'%(str(len(all_qubits))), qasm)
        for i in range(len(active_qubits)):
            idx=active_qubits[i]._index
            qasm = re.sub(r'\bq\[%s\]'%(str(idx)), 'TEMP_Q%s'%(str(idx)), qasm)

        qasm = re.sub(r'TEMP_Q%s'%(str(len(all_qubits))), 'q[%s]'%(str(len(QBIT_MAP))), qasm)
        for i in range(len(active_qubits)):
            idx=active_qubits[i]._index
            qasm = re.sub(r'TEMP_Q%s'%(str(idx)), mapping[idx], qasm)

        qasm=qasm.replace("\n","")       

        if re.search("delay.*", qasm):
            raise QmioException("Delay instruction is not supported in OpenQASM 2.0. Please, although could be slower, user option 'output_qasm3' to run this circuit.")
        return qasm
    
    def _to_openpulse(self,c):
        if self._exporter==None:
            self._exporter=OPExporter(logging_level=self._logger.level)
        return self._exporter.dumps(c).replace("\n","")
    
    def run(self, run_input: Union[Union[QuantumCircuit,Schedule,ScheduleBlock, str],List[Union[QuantumCircuit,Schedule,str]]], **options) -> QmioJob:
        """Run on QMIO QPU. This method is Synchronous, so it will wait for the results from the QPU
        
        
        Args:
            run_input (QuantumCircuit or Schedule or Str - a QASM string - or list): An
                individual or a list of :class:`~qiskit.circuit.QuantumCircuit`,
                or :class:`~qiskit.pulse.Schedule` or a string with a QASM 2.0/3.0 objects to
                run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object. Default options can be obtained with :meth:`_default_options`
                
                Some specific options for Qmio are:
                
                * memory: if format is binary_counts (the defualt format), returns also all the sequence.
                * repetition_period, slot of time between shot starts (default, **None**. Uses the default that it is calibrated)
                * res_format, format for the output (default, 'binary_count'. You can get the possible formats with :meth:`formats`)
                * output_qasm3, if convert the QuantumCircuit to OpenQASM 3.0 instead of OpenQASM 2.0 - default-)
                
                
        .. attention::
                When memory option is specified for binary_counts, take into account that the process of converting from raw data to shots is done by this class and not by Qmio QPU. 
                The split is done using values less than 0 or grand than 0.

        Returns:
            :class:`QmioJob`: The job object for the run
        
        Raises:
            QPUException: if there is one error in the QPU
            QmioException: if there are errors in the input parameters.
        """

        if isinstance(options,Options):
            shots=options.get("shots",default=self._options.get("shots"))
            memory=options.get("memory",default=self._options.get("memory"))
            repetition_period=options.get("repetition_period",default=self._options.get("repetition_period"))
            res_format=options.get("res_format",default=self._options.get("res_format"))
            output_qasm3=options.get("output_qasm3",default=self._options.get("output_qasm3"))
        else:
            if "shots" in options:
                shots=options["shots"]
            else:
                shots=self._options.get("shots")
            
            if "memory" in options:
                memory=options["memory"]
            else:
                memory=self._options.get("memory")

            if "repetition_period" in options:
                repetition_period=options["repetition_period"]
            else:
                repetition_period=self._options.get("repetition_period")
            
            if "res_format" in options:
                res_format=options["res_format"]
            else:
                res_format=self._options.get("res_format")
                
            if "output_qasm3" in options:
                output_qasm3=options["output_qasm3"]
            else:
                output_qasm3=self._options.get("output_qasm3")
            

        
        self._logger.info("Requested parameters: Shots %d - memory %s - Repetition_period %s - Res_format %s"%(shots, memory, str(repetition_period), res_format))
               
        if res_format not in FORMATS:
            raise QmioException("Format %s not in available formats:%s"%(res_format,FORMATS))

        if isinstance(run_input,str) and not "OPENQASM" in run_input:
            raise QmioException("Input seems not to be a valid OPENQASM 3.0 file...")
        
        if isinstance(run_input,QuantumCircuit) or isinstance(run_input,Schedule) or isinstance(run_input,ScheduleBlock) or isinstance(run_input,str):
            circuits=[run_input]
        else:
            circuits=run_input

        if shots*len(circuits) > self.max_shots:
            raise QmioException("Total number of shots %d larger than capacity %d"%(shots,self.max_shots))
        
        #self._logger.debug("Starting QmioRuntimeService")
        #service = QmioRuntimeService()
        
        if self._QPUBackend is None:
            self._logger.debug("Starting backend")
            self.connect()
                          
        
        job_id=uuid.uuid4()
                          
        self._logger.debug("Job id %s"%job_id)
                          
        ExpResult=[]
        
        for circuit in circuits:

            if isinstance(circuit,QuantumCircuit):
                if len(circuit.cregs)>1:
                    c=FlattenCircuit(circuit)
                else:
                    c=circuit
            else:
                c=circuit

            #print("Metadata",c.metadata)
            if isinstance(c,QuantumCircuit):
                try:
                    if output_qasm3:
                        qasm=self._to_qasm3(c)
                    else:
                        qasm=self._to_qasm2(c)
                except:
                    try:
                        qasm=self._to_openpulse(c)
                    except:
                        raise QmioException("Error converting circuit: %s"%c.name)
                #print(qasm)
            elif isinstance(c,Schedule) or isinstance(c,ScheduleBlock):
                qasm=self._to_openpulse(c)
                
            else:
                qasm=c

            # parche
            #self._logger.info("Replacing SC gate by RX(pi/2) as a temporal fix")
            #qasm=qasm.replace("SX ","rx(pi/2) ").replace("sx ","rx(pi/2) ")
            #self._logger.debug("Final submitted circuit %s"%qasm)
            
            remain_shots=shots
            ExpDict={}
            ExpList=[]
            self._logger.info("QASM to execute %s"%qasm)
            warning_raised = False
            while (remain_shots > 0):
                self._logger.info("Requesting SHOTS=%d"%min(self._max_shots,remain_shots))
                if memory:
                    _res_format="raw"
                else:
                    _res_format= res_format
                results = self._QPUBackend.run(circuit=qasm, shots=min(self._max_shots,remain_shots),repetition_period=repetition_period,res_format=_res_format)

                self._logger.debug("Results:%s"%results)
                if "Exception" in results:
                    raise QPUException(results["Exception"])
                
                try:
                    r=results["results"][list(results["results"].keys())[0]]
                except:
                    if res_format == "raw":
                        ExpList.append(results["results"])
                    else:
                        raise QPUException("QPU does not return results")
                
                if res_format== "binary_count":
                    if not memory:
                        for k in r:
                            key=hex(int(k[::-1],base=2))
                            ExpDict[key]=ExpDict[key]+r[k] if key in ExpDict else r[k]
                    else:
                        self._logger.debug("Output of type %s in memory register"%res_format)
                        a=np.array(r)
                        b=(a<0).astype(int).astype(str)
                        for i in range(b.shape[1]):
                            s=""
                            for j in b[:,i][::-1]: s=s+j
                            key=hex(int(s,base=2))
                            ExpDict[key]=ExpDict[key]+1 if key in ExpDict else 1
                            ExpList.append(key)
                            #self._logger.debug("QmioException - Binary for the next version- ")
                            #raise QmioException("Binary for the next version")
                else:
                    self._logger.debug("Output of type %s in memory register"%res_format)
                    try:
                        s=ExpList[0].copy()
                        for k in len(s):
                            s[k].append(r[k])
                    except:
                        s=r.copy()
                    ExpList=s
                    
                    
                remain_shots=remain_shots-self._max_shots
            
            
            
            if isinstance(c,QuantumCircuit):
                metadata=c.metadata
            else:
                metadata={}

            if "execution_metrics" in results:
                metadata["execution_metrics"]=results["execution_metrics"]

            metadata["repetition_period"]=repetition_period
            metadata["res_format"]=res_format

            creg_sizes=[]
            qreg_sizes=[]
            memory_slots=0
            n_qubits=0
            if isinstance(circuit,str):
                c_copy=circuit
                circuit=QasmCircuit()
                circuit.circuit=c_copy
                circuit.name="QASM"
                c=circuit
            elif isinstance(c,Schedule) or isinstance(c,ScheduleBlock):
                pointer=0
                try:
                    while len(qasm)>0:
                        b=qasm[pointer:].index("bit[")+4
                        q=qasm[pointer+b:].index("]")
                        
                        size=int(qasm[pointer+b:pointer+b+q])
                        p=qasm[pointer+b+q:].index(";")
                        name=qasm[pointer+b+q+1:pointer+b+q+p].strip()
                        pointer=pointer+b+p+1
                        
                        creg_sizes.append([name,size])
                        memory_slots+=size
                except:
                    pass

                try:
                    for c1 in circuit.qregs:
                        qreg_sizes.append([c1.name,c1.size])
                except:
                    qreg_sizes=creg_sizes.copy()
                try:
                    n_qubits=len(circuit.qubits)
                except:
                    n_qubits=memory_slots
            else:
                for c1 in circuit.cregs:
                    creg_sizes.append([c1.name,c1.size])
                    memory_slots+=c1.size

                for c1 in circuit.qregs:
                    qreg_sizes.append([c1.name,c1.size])
                n_qubits=len(circuit.qubits)
            header ={'name': c.name, 'creg_sizes':creg_sizes, 'memory_slots':memory_slots, 'n_qubits':n_qubits,
                           'qreg_sizes':qreg_sizes,'metadata':circuit.metadata}
            #header.update(c.metadata)

            self._logger.debug("Retorno counts: %s"%ExpDict)
            self._logger.debug("Returning memory: %s"%ExpList)
            
            dd={
                'shots': shots,
                'success': True,
                'header': header,
                }
            if (res_format != "raw") and not memory:
                dd['data']={
                    'counts': ExpDict,
                    'metadata': metadata,
                }
            else:
                dd['data']={
                    'counts': ExpDict,
                    'memory': ExpList,
                    'metadata': metadata,
                }
            ExpResult.append(dd)

        result_dict = {
            'backend_name': self._name,
            'backend_version': self._version,
            'qobj_id': None,
            'job_id': job_id,
            'success': True,
            'results': ExpResult,
            'date': datetime.now().isoformat(),

        }
        self._logger.debug("Final Results returned: %s"%result_dict)
                          
        results=Result.from_dict(result_dict)

        job=QmioJob(backend=self,job_id=uuid.uuid4(), jobstatus=JobStatus.DONE, result=results)
       
        return job
    
    
    
