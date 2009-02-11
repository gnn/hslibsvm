{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- | This module contains the translation of LibSVM's C interface to haskell.
-- It is mostly a literal translation, so anyone who wants to build his own
-- high level interface to LibSVM can do this using this module.
-- For every exported entity the name of the corresponding C entity is included
-- in the documentation and when translating C identifiers into haskell ones,
-- most times only the @svm_@ prefixes have been dropped since it is basically 
-- a job of the module system to take care of such things.
--------------------------------------------------------------------------------

module Data.Datamining.Classification.LibSVM.C (
  -- * Types
  -- | Translations of C structs and enums.

  -- ** Supported SVM Variants
  SVMType, c_svc, nu_svc, one_class, epsilon_svr, nu_svr

  -- ** Supported Kernel Functions
, KernelFunction
, linear, polynomial, poly, radialBasisFunction, rbf, sigmoid, precomputed

  -- ** Training Input
  -- *** LibSVM's Vector Representation
  -- | LibSVM represents it's input vectors as sparse vectors with entries
  -- of type @double@. To represent sparsity, values of type @double@ are 
  -- paired with their indices and missing indices are treated as zero
  -- values. Indices have to be in ascending order and the each input vector
  -- has to be terminated with a node containing an index of -1. 
, Node(..), NodeP, NodePP

  -- *** Problem Formulation
, Problem(..), ProblemP

  -- *** Training Parameters
, Parameters(..), ParametersP

  -- ** Training output
, Model

  -- * Functions
  -- | The documentation to these functions has been copied from LibSVM's
  -- README file with a few minor changes.

  -- ** Training
, cross_validation 
, train   

  -- ** Serialization
, load_model
, save_model

  -- ** Model Querying 
, check_probability_model
, get_labels
, get_nr_class
, get_svm_type
, get_svr_probability

  -- ** Prediction
, predict
, predict_values
, predict_probability

  -- ** Memory Management
, destroy_model
, destroy_parameters

  -- ** Sanity Checking
, check_parameters
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
-- Boilerplate for hsc2hs
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

-- | Type safe version of the C @enum@eration containing the constants used 
-- to select the SVM variant to train. Use the exported 
-- values of type @'SVMType'@ to control which type of svm is going to be 
-- trained by setting the @'svm_type'@ field to the appropriate value.
-- This way the type checker makes sure the @'svm_type'@ field of 
-- @'Parameters'@ is only set to allowed values.

newtype SVMType = SVMType {unSVM :: CInt}

#{enum SVMType, SVMType
, c_svc       = C_SVC
, nu_svc      = NU_SVC
, one_class   = ONE_CLASS
, epsilon_svr = EPSILON_SVR
, nu_svr      = NU_SVR
}

-- | Type safe version of the C @enum@eration containing the constants used
-- to choose the kernel function to be used in training and prediction.
-- Again the exported values of type @'KernelFunction'@ should be used to set
-- the @'kernel_type'@ field to the appropriate value.

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


-- | Sparse representation of input vectors. 
-- 
-- C Type: @struct svm_node;@  
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

-- | This is the way LibSVM bundles input vectors and corresponding classes
-- together as input for training the support vector machine.
--
-- C Type:  @struct svm_problem;@
data Problem = Problem { 
  -- | The number of training samples.
  size :: CInt
, -- | The labels. For regression this would be real numbers while for 
  -- classificiation this should be integers.
  labels :: Ptr CDouble 
, -- | The array of input vectors.
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

-- | LibSVM's way of passing the various training parameters to the 
-- support vector machine algorithm.
--
-- C Type: @struct svm_parameter;@
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
  -- two fields, just set @'weight_labels'@ to 0.
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

-- | This function constructs and returns an SVM model according to
-- the given training data and parameters.
--
-- C declaration: @struct svm_model *svm_train(const struct svm_problem *, 
--  const struct svm_parameter *);@
foreign import ccall unsafe "svm.h svm_train"
  train :: ProblemP -> ParametersP -> IO Model

-- | This function conducts cross validation.
-- @cross_validation prob param nr_fold target@ separates data into
-- nr_fold folds. Under given parameters, sequentially each fold is
-- validated using the model from training the remaining. Predicted
-- labels (of all @prob@'s instances) in the validation process are
-- stored in the array called @target@.
--
-- C declaration: @void svm_cross_validation(const struct svm_problem *,
--  const struct svm_parameter *, int, double *);@
foreign import ccall unsafe "svm.h  svm_cross_validation"
  cross_validation :: ProblemP -> ParametersP -> CInt -> Ptr CDouble -> IO ()

-- | This function saves a model to a file; returns 0 on success, or -1
-- if an error occurs.
--
-- C declaration: @int svm_save_model(const char *,const struct svm_model *);@
foreign import ccall unsafe "svm.h svm_save_model"
  save_model :: CString -> Model -> IO CInt

-- | This function returns a pointer to the model read from the file,
-- or a null pointer if the model could not be loaded.
--
-- C declaration: @struct svm_model *svm_load_model(const char *);@
foreign import ccall unsafe "svm.h svm_load_model"
  load_model :: CString -> IO Model

-- | This function gives svm_type of the model. Possible values of
-- svm_type are defined in svm.h.
--
-- C declaration: @int svm_get_svm_type(const struct svm_model *);@
foreign import ccall unsafe "svm.h svm_get_svm_type"
  get_svm_type :: Model -> IO SVMType

-- | For a classification model, this function gives the number of
-- classes. For a regression or an one-class model, 2 is returned.
--
-- C declaration @int svm_get_nr_class(const svm_model *);@
foreign import ccall unsafe "svm.h svm_get_nr_class"
  get_nr_class :: Model -> IO CInt

-- | For a classification model, @get_labels model label@ outputs the name of
-- labels into the array @label@. For regression and one-class
-- models, @'label'@ is unchanged.
--
-- C declaration: @void svm_get_labels(const svm_model *, int*);@
foreign import ccall unsafe "svm.h svm_get_labels"
  get_labels :: Model -> Ptr CInt -> IO ()

-- | For a regression model with probability information, this function
-- outputs a value sigma > 0. For test data, we consider the
-- probability model: target value = predicted value + z, z: Laplace
-- distribution e^(-|z|/sigma)/(2sigma)

-- If the model is not for svr or does not contain required
-- information, 0 is returned.

-- C declaration: @double svm_get_svr_probability(const struct svm_model *);@
foreign import ccall unsafe "svm.h svm_get_svr_probability"
  get_svr_probability :: Model -> IO CDouble

-- | This function gives decision values on a test vector x given a
-- model.
-- 
-- For a classification model with nr_class classes, 
-- @predict_values model x dec_values@
-- gives nr_class*(nr_class-1)/2 decision values in the array
-- @dec_values@, where nr_class can be obtained from the function
-- @'get_nr_class'@. The order is label[0] vs. label[1], ...,
-- label[0] vs. label[nr_class-1], label[1] vs. label[2], ...,
-- label[nr_class-2] vs. label[nr_class-1], where label can be
-- obtained from the function @'get_labels'@.
-- 
-- For a regression model, label[0] is the function value of @x@
-- calculated using the model. For one-class model, label[0] is +1 or
-- -1.

-- C declaration: @void svm_predict_values(const svm_model *, 
--  const svm_node *, double*)@
foreign import ccall unsafe "svm.h svm_predict_values"
  predict_values :: Model -> NodeP -> Ptr CDouble -> IO ()

-- | @predict model x@ does classification or regression on a test vector 
-- @x@ given a @model@.
-- 
-- For a classification model, the predicted class for @x@ is returned.
-- For a regression model, the function value of @x@ calculated using
-- the model is returned. For an one-class model, +1 or -1 is
-- returned.
--
-- C declaration: @double svm_predict(const struct svm_model *, 
--  const struct svm_node *);@
foreign import ccall unsafe "svm.h svm_predict"
  predict :: Model -> NodeP -> IO CDouble

-- | @predict_probability model x prob_etimates@ does classification or 
-- regression on a test vector @x@ given a @model@ with probability 
-- information.
-- 
-- For a classification model with probability information, this
-- function gives nr_class probability estimates in the array
-- @prob_estimates@. nr_class can be obtained from the function
-- @'get_nr_class'@. The class with the highest probability is
-- returned. For regression/one-class SVM, the array @prob_estimates@
-- is unchanged and the returned value is the same as that of
-- @'predict'@.
--
-- C declaration: @double svm_predict_probability(const struct svm_model *,
--  const struct svm_node *, double*);@
foreign import ccall unsafe "svm.h svm_predict_probability"
  predict_probability :: Model -> NodeP -> Ptr CDouble -> IO CDouble

-- | This function frees the memory used by a model.
--
-- C declaration: @void svm_destroy_model(struct svm_model *);@
foreign import ccall unsafe "svm.h svm_destroy_model"
  destroy_model :: Model -> IO ()

-- | This function frees the memory used by a parameter set.
--
-- C declaration: @void svm_destroy_param(struct svm_parameter *);@
foreign import ccall unsafe "svm.h svm_destroy_param"
  destroy_parameters :: ParametersP -> IO ()

-- | This function checks whether the parameters are within the feasible
-- range of the problem. This function should be called before calling
-- @'train'@ and @'cross_validation'@. It returns NULL if the
-- parameters are feasible, otherwise an error message is returned.
--
-- C declaration: @const char *svm_check_parameter(const struct svm_problem *, 
--  const struct svm_parameter *);@
foreign import ccall unsafe "svm.h svm_check_parameter"
  check_parameters :: ProblemP -> ParametersP -> IO CString

-- | This function checks whether the model contains required
-- information to do probability estimates. If so, it returns
-- +1. Otherwise, 0 is returned. This function should be called
-- before calling @'get_svr_probability'@ and @'predict_probability'@.
--
-- C declaration: @int svm_check_probability_model(const struct svm_model *);@
foreign import ccall unsafe "svm.h svm_check_probability_model"
  check_probability_model :: Model -> IO CInt

