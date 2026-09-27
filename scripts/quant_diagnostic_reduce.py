"""Optional fused FP64 metric reducer; plain NumPy remains the default."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import numpy as np

NAMES=['input_only','weight_only','int8_pair','ideal_int10','actual_requant',
       'final_local_total','ideal_requantization_only','requantization_only']


class NativeMetrics:
    def __init__(self):
        source=Path(__file__).with_name('quant_metric_reduce.c')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        target=Path(tempfile.gettempdir())/('quant-metric-'+digest+'.dylib')
        if not target.exists():
            subprocess.run(['cc','-O3','-std=c11','-ffp-contract=off','-shared',str(source),'-o',str(target)],check=True)
        self.library=ctypes.CDLL(str(target))
        self.function=self.library.reduce_metrics
        ptr=ctypes.POINTER(ctypes.c_double)
        self.function.argtypes=[ctypes.c_size_t]+[ptr]*7
        self.function.restype=ctypes.c_int
        self.ptr=ptr

    def add(self,metrics,reference,input_only,weight_only,pair,ideal,hardware):
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in [reference,input_only,weight_only,pair,ideal,hardware]]
        if any(a.shape!=arrays[0].shape for a in arrays):raise ValueError('metric shape mismatch')
        result=np.zeros((8,4),np.float64)
        status=self.function(arrays[0].size,*[a.ctypes.data_as(self.ptr) for a in arrays],result.ctypes.data_as(self.ptr))
        if status or not np.isfinite(result).all():raise ValueError('nonfinite fused metric')
        for name,values in zip(NAMES,result):
            if name not in metrics:
                continue
            metric=metrics[name];metric.count+=arrays[0].size
            metric.square+=float(values[0]);metric.absolute+=float(values[1]);metric.energy+=float(values[2])
            metric.maximum=max(metric.maximum,float(values[3]))
