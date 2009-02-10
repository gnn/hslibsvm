{-# LANGUAGE ForeignFunctionInterface #-}
-- | This module contains the translation of LibSVM's C interface to haskell.
-- It is mostly a literal translation. I only dropped the @svm_@ prefixes and
-- translated a few types into the corresponding haskell ones where 
-- appropriate.
module Data.Datamining.Classification.LibSVM.C (
  SVMType, c_svc, nu_svc, one_class, epsilon_svr, nu_svr
, KernelFunction
, linear, polynomial, poly, radialBasisFunction, rbf, sigmoid, precomputed
, Node(..), NodeP, NodePP
, Problem(..), ProblemP
, Parameters(..), ParametersP
, Model
, train   
, cross_validation
, save_model
, load_model
, get_svm_type
, get_nr_class
, get_labels
, get_svr_probability
, predict_values
, predict
, predict_probability
, destroy_model
, destroy_parameters
, check_parameters
, check_probability_model
) where

--------------------------------------------------------------------------------
-- Standard Modules
--------------------------------------------------------------------------------

import Foreign
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Utils

--------------------------------------------------------------------------------
-- Private Modules
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Boilerplate: hsc2hs
--------------------------------------------------------------------------------

-- This include us to get access to @offsetof@.

#include <stddef.h>

-- This macro is necessary to calculate ther correct alignment constraints
-- for a foreign type in a storable instance declaration.
#{let alignment t = 
  "%lu", (unsigned long)offsetof(struct {char x__; t(y__); }, y__)}

-- Next we need to include libsvm's main header file to get acces to the 
-- structs and prototypes available from the library

#include <svm.h>

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

-- | LibSVM supports a number of different SVM types. Use the exported 
-- values of type 'SVMType' to set which type of svm is going to be trained.

newtype SVMType = SVMType {unSVM :: CInt}

#{enum SVMType, SVMType
, c_svc       = C_SVC
, nu_svc      = NU_SVC
, one_class   = ONE_CLASS
, epsilon_svr = EPSILON_SVR
, nu_svr      = NU_SVR
}

-- | LibSVM also supports a number of different types of kernel functions.
-- Again the exported values of type 'KernelFunction' should be used to set
-- the type one wants to use during training.

newtype KernelFunction = KernelFunction {unKernel :: CInt}

#{enum KernelFunction, KernelFunction
, linear              = LINEAR
, polynomial          = POLY
, poly                = POLY
, radialBasisFunction = RBF
, rbf                 = RBF
, sigmoid             = SIGMOID
, precomputed         = PRECOMPUTED
}

-- This comment, the next one and the empty line between them seem 

-- necessary so that haddock puts the comment for the next data 
-- declaration in the right place.

-- | LibSVM represents it's input vectors as sparse vectors with entries
-- of type @double@. To represent sparsity values of type @double@ are 
-- paired with their indices and missing indices are treated as zero
-- values. Indices have to be in ascending order and the each input vector
-- has to be terminated with a node containing an index of @-1@. 
-- The type @Node@ is just the translation of @svm.h@'s @struct svm_node@ 
-- into a haskell representation.  
data Node = Node {
  index :: CInt
, value :: CDouble
}

instance Storable Node where
  sizeOf _    = (#size struct svm_node)
  alignment _ = (#alignment struct svm_node)
  peek p      = do
    i <- (#peek struct svm_node, index) p
    v <- (#peek struct svm_node, value) p
    return $! Node i v
  poke p (Node i v) = 
    (#poke struct svm_node, index) p i >>
    (#poke struct svm_node, value) p v 
    
type NodeP = Ptr Node

type NodePP = Ptr NodeP

-- | This is the way LibSVM saves training input. 
data Problem = Problem { 
  -- | The number of training samples.
  size :: CInt
, -- | The labels. For regression this would be real numbers while for 
  -- classificiation this should be integers.
  labels :: Ptr CDouble 
, -- The array of input vectors.
  inputs :: NodePP
}

instance Storable Problem where
  sizeOf _    = (#size struct svm_problem)
  alignment _ = (#alignment struct svm_problem)
  peek p      = do
    size    <- (#peek struct svm_problem, l) p
    labels  <- (#peek struct svm_problem, y) p
    inputs  <- (#peek struct svm_problem, x) p
    return $! Problem size labels inputs
  poke p (Problem s l is) =
    (#poke struct svm_problem, l) p s >>
    (#poke struct svm_problem, y) p l >>
    (#poke struct svm_problem, x) p is

type ProblemP = Ptr Problem

-- | The type @Parameters@ is the haskell translation of the C type
-- @struct svm_paramter@. This is LibSVM's way of passing the various
-- training parameters to the support vector machine algorithm.
data Parameters = Parameters {
  -- | The type of support vector machine to train.
  svm_type :: SVMType
, -- | The type of kernel function to use.
  kernel_type :: KernelFunction
, -- | This is only used for 'polynomial' kernel functions.
  degree :: CInt
, -- | The value of @gamma@ for 'polynomial', 'rbf' or 'sigmoid' kernel
  -- functions.
  gamma :: CDouble
, -- | The added constant to use in the 'polynomial' and the 'sigmoid'
  -- kernel functions
  c0  :: CDouble
, -- | The cache size to use during training specified in MB.
  cache_size :: CDouble
, -- | The stopping criterion.
  epsilon :: CDouble
, -- | The C paramter for the SVM types 'c_svc', 'epsilon_svr' and 'nu_svr'.
  parameterC :: CDouble
, -- | The following three fields, namely @'weight_labels'@, @'weight_label'@, 
  -- and @'weight'@ are used to change the penalty for some classes. If the 
  -- weight for a class is not changed, it is set to 1. 
  -- This is useful for training classifier using unbalanced input data or 
  -- with asymmetric misclassification cost.
  -- This field contains the number of weight labels and weights, i.e. 
  -- the number of entries in @'weight_label'@ and @'weight'@. 
  -- If you don't want to use the feature provided by this and the next 
  -- two fields, just set @'weight_labels'@ to @0@.
  weight_labels :: CInt
, -- | The weight labels matching the given weights.
  weight_label :: Ptr CInt
, -- | The weights for each label given above. Each @weight[i]@ corresponds
  -- to @weight_label[i]@, meaning that the penalty of class 
  -- @weight_label[i]@ is scaled by a factor of @weight[i]@.
  weight :: Ptr CDouble
, -- | The parameter @nu@ used for the SVM types @'nu_svc'@, @'nu_svr'@, and 
  -- @'one_class'@. It approximates the fraction of training errors and 
  -- support vectors.
  nu :: CDouble
, -- | This is the epsilon in epsilon-insensitive loss function of 
  -- epsilon-SVM regression, i.e. SVM type @'epsilon_svr'@.
  epsilon' :: CDouble
, -- | This flag decides whether to use the shrinking heuristics or not.
  -- If set to 'True' shrinking is turned on, otherwise it's turned off.
  shrinking :: CInt
, -- | This flag decides whether to obtain a model with probability 
  -- information or not. Probability information is gathered if and only 
  -- if this flag is set to 'True'.
  probability :: CInt
}

instance Storable Parameters where
  sizeOf    _ = (#size struct svm_parameter)
  alignment _ = (#alignment struct svm_parameter)
  peek p      = do
    svm     <- (#peek struct svm_parameter, svm_type) p >>= return . SVMType
    kernel  <- (#peek struct svm_parameter, kernel_type) p >>= 
      return . KernelFunction
    degree  <- (#peek struct svm_parameter, degree) p
    gamma   <- (#peek struct svm_parameter, gamma) p
    c0      <- (#peek struct svm_parameter, coef0) p
    cache   <- (#peek struct svm_parameter, cache_size) p
    eps     <- (#peek struct svm_parameter, eps) p
    c       <- (#peek struct svm_parameter, C) p
    wlsCnt  <- (#peek struct svm_parameter, nr_weight) p
    wls     <- (#peek struct svm_parameter, weight_label) p
    ws      <- (#peek struct svm_parameter, weight) p
    nu      <- (#peek struct svm_parameter, nu) p
    eps'    <- (#peek struct svm_parameter, p) p
    shrink  <- (#peek struct svm_parameter, shrinking) p >>= return
    probs   <- (#peek struct svm_parameter, probability) p >>= return
    return $! 
      Parameters 
        svm kernel degree gamma c0 cache eps c wlsCnt 
        wls ws nu eps' shrink probs
  poke p ps =
    ((#poke struct svm_parameter, svm_type) p . unSVM) (svm_type ps) >>
    ((#poke struct svm_parameter, kernel_type) p . unKernel) (kernel_type ps) 
    >>
    ((#poke struct svm_parameter, degree) p) (degree ps) >>
    ((#poke struct svm_parameter, gamma) p) (gamma ps) >>
    ((#poke struct svm_parameter, coef0) p) (c0 ps) >>
    ((#poke struct svm_parameter, cache_size) p) (cache_size ps) >>
    ((#poke struct svm_parameter, eps) p) (epsilon ps) >>
    ((#poke struct svm_parameter, C) p) (parameterC ps) >>
    ((#poke struct svm_parameter, nr_weight) p) (weight_labels ps) >>
    ((#poke struct svm_parameter, weight_label) p) (weight_label ps) >>
    ((#poke struct svm_parameter, weight) p) (weight ps) >>
    ((#poke struct svm_parameter, nu) p) (nu ps) >>
    ((#poke struct svm_parameter, p) p) (epsilon' ps) >>
    ((#poke struct svm_parameter, shrinking) p) (shrinking ps) >>
    ((#poke struct svm_parameter, probability) p) (probability ps)

type ParametersP = Ptr Parameters

-- | This is the model structure used by LibSVM. Since it is not exportet 
-- by the header @svm.h@ it is treated as and opaque structure and only
-- meant to be passed around between functions expecting it as a parameter.
-- The only way of obtaining (a pointer to) a value of this type is through
-- the C functions @svm_train@ and @svm_load_model@.
type Model = Ptr ()

--------------------------------------------------------------------------------
-- Foreign Imports
--------------------------------------------------------------------------------

foreign import ccall unsafe "svm.h svm_train"
  train :: ProblemP -> ParametersP -> IO Model

foreign import ccall unsafe "svm.h  svm_cross_validation"
  cross_validation :: 
    ProblemP -> ParametersP -> CInt -> Ptr CDouble -> IO ()

foreign import ccall unsafe "svm.h svm_save_model"
  save_model :: CString -> Model -> IO CInt

foreign import ccall unsafe "svm.h svm_load_model"
  load_model :: CString -> IO Model

foreign import ccall unsafe "svm.h svm_get_svm_type"
  get_svm_type :: Model -> IO SVMType

foreign import ccall unsafe "svm.h svm_get_nr_class"
  get_nr_class :: Model -> IO CInt

foreign import ccall unsafe "svm.h svm_get_labels"
  get_labels :: Model -> Ptr CInt -> IO ()

foreign import ccall unsafe "svm.h svm_get_svr_probability"
  get_svr_probability :: Model -> IO CDouble

foreign import ccall unsafe "svm.h svm_predict_values"
  predict_values :: Model -> NodeP -> Ptr CDouble -> IO ()

foreign import ccall unsafe "svm.h svm_predict"
  predict :: Model -> NodeP -> IO CDouble

foreign import ccall unsafe "svm.h svm_predict_probability"
  predict_probability :: Model -> NodeP -> Ptr CDouble -> IO CDouble

foreign import ccall unsafe "svm.h svm_destroy_model"
  destroy_model :: Model -> IO ()

foreign import ccall unsafe "svm.h svm_destroy_param"
  destroy_parameters :: ParametersP -> IO ()

foreign import ccall unsafe "svm.h svm_check_parameter"
  check_parameters :: ProblemP -> ParametersP -> IO CString

foreign import ccall unsafe "svm.h svm_check_probability_model"
  check_probability_model :: Model -> IO CInt

