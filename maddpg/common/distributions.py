import tensorflow as tf
import numpy as np
import maddpg.common.tf_util as U
from tensorflow.python.ops import math_ops
from multiagent.multi_discrete import MultiDiscrete
from tensorflow.python.ops import nn

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return SoftCategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return [self.ncat]
    def sample_dtype(self):
        return tf.float32

class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class SoftMultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return SoftMultiCategoricalPd
    def pdfromflat(self, flat):
        return SoftMultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [sum(self.ncats)]
    def sample_dtype(self):
        return tf.float32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32

# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.argmax(self.logits, axis=1)
    def logp(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.softmax(self.logits, axis=-1)
    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)  
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)        

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.int32)
        self.categoricals = list(map(CategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def logp(self, x):
        return tf.add_n([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):  # doesn't work yet
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.float32)
        self.categoricals = list(map(SoftCategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].mode())
        return tf.concat(x, axis=-1)
    def logp(self, x):
        return tf.add_n([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].sample())
        return tf.concat(x, axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat        
    def mode(self):
        return self.mean
    def logp(self, x):
        return - 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=1) \
               - 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) \
               - U.sum(self.logstd, axis=1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=1)
    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), 1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.round(self.ps)
    def logp(self, x):
        return - U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=1)
    def kl(self, other):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=1) - U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def entropy(self):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def sample(self):
        p = tf.sigmoid(self.logits)
        u = tf.random_uniform(tf.shape(p))
        return tf.to_float(math_ops.less(u, p))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        # return CategoricalPdType(ac_space.n)
        return SoftCategoricalPdType(ac_space.n)
    elif isinstance(ac_space, MultiDiscrete):
        #return MultiCategoricalPdType(ac_space.low, ac_space.high)
        return SoftMultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError

def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]
