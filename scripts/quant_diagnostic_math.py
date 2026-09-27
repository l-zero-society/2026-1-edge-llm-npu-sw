"""Independent counterfactual metrics; never add MSEs across paths."""
import numpy as np
from static_quant.core import exact_accumulator, quantize_input
from static_quant.hardware import finite, positive_scale

_dgemm = None


def configure_blas(backend):
    """Optional macOS FP64 BLAS without changing the historical NumPy runtime."""
    global _dgemm
    if backend == 'numpy':
        _dgemm = None
    elif backend == 'accelerate':
        import ctypes
        import sys
        if sys.platform != 'darwin':
            raise ValueError('Accelerate is only available on macOS')
        lib = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        _dgemm = lib.cblas_dgemm
        i, d, ptr = ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
        _dgemm.argtypes = [i, i, i, i, i, i, d, ptr, i, ptr, i, d, ptr, i]
        _dgemm.restype = None
    else:
        raise ValueError('unknown BLAS backend')


def dot(x, w):
    """Return FP64 X @ W.T with explicitly FP64 inputs and output."""
    x, w = np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError('expected X[M,K], W[N,K]')
    if _dgemm is None:
        return x @ w.T
    import ctypes
    ptr = ctypes.POINTER(ctypes.c_double)
    m, k = x.shape
    n = len(w)
    out = np.empty((m, n), dtype=np.float64)
    _dgemm(101, 111, 112, m, n, k, 1., x.ctypes.data_as(ptr), k,
           w.ctypes.data_as(ptr), k, 0., out.ctypes.data_as(ptr), n)
    return out


class ErrorMetric:
    def __init__(self):
        self.count = 0
        self.square = self.absolute = self.energy = self.maximum = 0.0

    def add(self, value, reference):
        value = finite(np.asarray(value, dtype=np.float64), 'metric value')
        reference = finite(np.asarray(reference, dtype=np.float64), 'metric reference')
        if value.shape != reference.shape:
            raise ValueError('metric shape mismatch')
        error = value-reference
        self.count += error.size
        self.square += float(np.square(error).sum())
        self.absolute += float(np.abs(error).sum())
        self.energy += float(np.square(reference).sum())
        self.maximum = max(self.maximum, float(np.max(np.abs(error), initial=0)))

    def result(self):
        if not self.count:
            raise ValueError('empty metric')
        return dict(mse=self.square/self.count, mae=self.absolute/self.count,
                    nmse=self.square/self.energy if self.energy else (0.0 if not self.square else None),
                    max_absolute_error=self.maximum, reference_energy=self.energy/self.count)


def integer_dot(xq, wq, backend='int64'):
    """FP64 acceleration is exact for bounded integers, checked against INT64.

    All products and every possible partial sum have absolute bound K*127**2
    below 2**31, hence below FP64's consecutive-integer limit 2**53.
    No noninteger operand or reduced-precision GEMM is used.
    """
    if backend == 'int64':
        return exact_accumulator(xq, wq)
    if backend != 'fp64-exact':
        raise ValueError('unknown integer backend')
    for q in (xq, wq):
        if q.dtype.kind not in 'iu' or np.any(np.abs(q.astype(np.int64)) > 127):
            raise ValueError('expected signed INT8 [-127,127]')
    if xq.shape[1]*127**2 > 2**31-1:
        raise OverflowError('cannot guarantee all INT32 prefixes')
    out = dot(xq, wq)
    if not np.array_equal(out, np.rint(out)):
        raise ArithmeticError('noninteger accelerated accumulator')
    return out.astype(np.int64)


def paths(x, w, sx, sw, s10, profile, multiplier, shift, backend='int64'):
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    xq = quantize_input(x, sx)
    wq = np.clip(np.rint(w/sw[:, None]), -127,127).astype(np.int8)
    acc = integer_dot(xq,wq,backend)
    pair = acc*(sx*sw)
    ideal_raw = np.rint(acc*(sx*sw/s10))
    ideal = np.clip(ideal_raw,-512,511)*s10
    raw = profile.apply(acc,multiplier,shift,saturate=False)
    hw = np.clip(raw,-512,511)*s10
    return dict(reference=x@w.T, input_only=(xq.astype(np.float64)*sx)@w.T,
                weight_only=x@(wq.astype(np.float64)*sw[:,None]).T,
                int8_pair=pair, ideal_int10=ideal, hw=hw,
                int10_clipped=(raw < -512)|(raw > 511),
                ideal_int10_clipped=(ideal_raw < -512)|(ideal_raw > 511))


def scale_from_threshold(threshold, absmax):
    return float(threshold/127 if threshold > 0 else absmax/127 if absmax > 0 else 1)


def activation_stats(x, sx):
    """Exact percentiles (no reservoir); x is a single operation/split memmap."""
    positive_scale(sx,'s_X')
    flat=np.asarray(x).reshape(-1)
    if not flat.size:
        raise ValueError('empty activation')
    finite(flat,'activation')
    absolute=np.abs(flat)  # one FP32 temporary, released after partition
    ps=[50,90,95,99,99.5,99.9,99.95,99.99,100]
    percentiles=np.percentile(absolute,ps,overwrite_input=True)
    r={f'p{p:g}':float(v) for p,v in zip(ps,percentiles)}
    del absolute
    total=square=error_sq=abs_error=0.0
    minimum,maximum=float(flat.min()),float(flat.max())
    zeros=ones=small=sat=clipped=below=0
    for start in range(0,len(flat),1048576):
        a=flat[start:start+1048576].astype(np.float64)
        q=quantize_input(a,sx);aq=np.abs(q.astype(np.int16));err=q.astype(np.float64)*sx-a
        total+=float(a.sum());square+=float(np.square(a).sum())
        abs_error+=float(np.abs(err).sum());error_sq+=float(np.square(err).sum())
        zeros+=int((q==0).sum());ones+=int((aq==1).sum());small+=int((aq<=2).sum())
        sat+=int((aq==127).sum());clipped+=int((np.abs(a)>127*sx).sum());below+=int((np.abs(a)<sx/2).sum())
    count=len(flat);mean=total/count;amax=max(abs(minimum),abs(maximum))
    r.update(min=minimum,max=maximum,absmax=amax,mean=mean,std=float(np.sqrt(max(0,square/count-mean**2))),
             s_X=sx,count=count,zero_rate=zeros/count,one_rate=ones/count,abs_q_le_2_rate=small/count,
             saturation_rate=sat/count,clipping_rate=clipped/count,below_half_scale_rate=below/count,
             mean_absolute_quantization_error=abs_error/count,
             sqnr_db=10*float(np.log10(square/error_sq)) if square and error_sq else None,
             sqnr_status='finite' if square and error_sq else 'zero_error' if not error_sq else 'zero_signal')
    for p in [50,99,99.9]:
        v=r[f'p{p:g}'];r[f'absmax_over_p{p:g}']=amax/v if v else None
    return r
