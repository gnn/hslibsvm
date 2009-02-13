{-# LANGUAGE FlexibleInstances #-}
--------------------------------------------------------------------------------
-- | This module contains an attempt to provide a more convenient interface to
-- LibSVM than the one gained by just using the translation of the
-- C interface.
--------------------------------------------------------------------------------

module Data.Datamining.Classification.LibSVM(
  -- * Types
  -- ** Input Types
  -- *** Input Vectors
  InputVector, SVMInput, inputVector
  -- *** Labeled Input Vectors
, Label, LabeledInput, label, labelList
  -- *** Training Input
, TrainingInput, Trainable, trainingInput
  -- ** Support Vector Machine Types
, C.SVMType, cSVC, nuSVC, oneClass, epsilonSVR, nuSVR
  -- ** Kernel Function Types
, C.KernelFunction
, linear, polynomial, poly, radialBasisFunction, rbf, sigmoid, precomputed
  -- ** Training Parameters
, Parameters(..), defaultNu
  -- ** Model
, Model
  -- * SVM Functions
, train, load, save
) where

--------------------------------------------------------------------------------
-- Standard Modules
--------------------------------------------------------------------------------

import Control.Exception
import Control.Monad
import Data.List
import qualified Data.Map as Map
import qualified Data.IntMap as IMap
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Error
import Foreign.Marshal.Utils
import Foreign.Ptr
import System.IO

--------------------------------------------------------------------------------
-- Private Modules
--------------------------------------------------------------------------------

import qualified Data.Datamining.Classification.LibSVM.C as C

--------------------------------------------------------------------------------
-- Types and Type Classes
--------------------------------------------------------------------------------

-- | The type of input vectors (without the labels) used to communicate 
-- with LibSVM.
newtype InputVector = InputVector { unInputVector :: [Double]}

-- | The class of the types which act as viable input vectors to LibSVM.
-- For lists of types which are an instance of the type class @'Real'@
-- a default implementation is provided so that you can use lists of 
-- most numeric types out of the box.
-- NOTE: The default instance for @'Real'@ uses @'realToFrac'@ which is
-- broken. Read
-- <http://www.mail-archive.com/haskell-prime@haskell.org/msg00790.html>
-- or
-- <http://www.mail-archive.com/haskell-cafe@haskell.org/msg52603.html>
-- and be aware of the danger.
class SVMInput a where
  inputVector :: a -> InputVector

instance Real a => SVMInput [a] where
  inputVector = InputVector . map (realToFrac)

instance SVMInput LabeledInput where
  inputVector = getVector

-- | The type of labeled input vectors, i.e. an @'InputVector'@ with a 
-- corresponding @'Label'@.
data LabeledInput = LabeledInput { 
  getLabel :: Double, 
  getVector :: InputVector
}

-- | The type of input used to train a support vector machine.
type TrainingInput = [LabeledInput]

-- | The type of labels used by LibSVM. 
-- Internally LibSVM just uses @double@s as labels.
-- The intended meaning is that, for regression the label should be 
-- the input's target value, while for classification the label
-- should be an integer representing the class label.
type Label = Double

-- | Labels an input vector.
label :: (SVMInput i) => i -> Label -> LabeledInput
label = flip LabeledInput . inputVector

-- | Labels a list of input vectors. 
labelList :: (SVMInput i) => [i] -> Label -> [LabeledInput]
labelList is l = map (flip label l) is

-- | The class of types which can be interpreted as something a support
-- vector machine can be trained from.
class Trainable a where trainingInput :: a -> TrainingInput

instance (SVMInput i) => Trainable [(Label, [i])] where 
  trainingInput = concatMap $ (uncurry . flip) labelList

instance (SVMInput i) => Trainable (Map.Map Label [i]) where
  trainingInput = trainingInput . Map.assocs

instance (SVMInput i) => Trainable (IMap.IntMap [i]) where
  trainingInput = trainingInput . 
    map (\(k, e) -> ((fromIntegral k)::Label, e)) . 
    IMap.assocs

-- | C-SVM classification
cSVC :: C.SVMType
cSVC = C.c_svc

-- | nu-SVM classification
nuSVC :: C.SVMType
nuSVC = C.nu_svc

-- | one-class-SVM
oneClass :: C.SVMType
oneClass = C.one_class

-- | epsilon-SVM regression
epsilonSVR :: C.SVMType
epsilonSVR = C.epsilon_svr

-- | nu-SVM regression
nuSVR :: C.SVMType
nuSVR = C.nu_svr

-- | The linear kernel function, i.e.:
--
-- * @K(x,y) = \<x,y\>@
--
-- where @\<x,y\>@ is the ordinary dot product between x and y.
linear :: C.KernelFunction
linear = C.linear

-- | Polynomial kernel function with the formula:
--
-- * @K(x,y) = (gamma*\<x,y\>+c0)^d@
--
-- where @\<x,y\>@ is the ordinary dot product between x and y.
polynomial :: C.KernelFunction
polynomial = C.polynomial

-- | Just a short name for the @'polynomial'@ kernel.
poly :: C.KernelFunction
poly = C.poly

-- | Radial basis function kernel having the formula:
--
-- * @K(x,y) = exp(-gamma*|x-y|^2)@
--
radialBasisFunction :: C.KernelFunction
radialBasisFunction = C.radialBasisFunction

-- | Just a shorter name for the @'radialBasisFunction'@ kernel.
rbf :: C.KernelFunction
rbf = C.rbf

-- | Sigmoid kernel. Formula:
--
-- * @K(x,y) = tanh(gamma*\<x,y\>+c0)@
--
-- where @\<x,y\>@ is the ordinary dot product between x and y.
-- Note that this kernel type does not yield a valid kernel function 
-- for some values of @gamma@ and @c0@.
sigmoid :: C.KernelFunction
sigmoid = C.sigmoid

-- | This means that the values of the kernel function are precomputed and 
-- can be found in the file containing the training data. 
-- No check is performed whether the resulting matrix of kernel function
-- values is valid.
precomputed :: C.KernelFunction
precomputed = C.precomputed

-- | LibSVM's way of passing the various training parameters 
-- to the support vector machine. This version uses 
-- ordinary haskell types as the field types. 
-- All the fields have the 
-- same meaning as the fields in the lower level translation of the 
-- C struct @'C.Parameters'@.
data Parameters = Parameters {
  -- | The type of support vector machine to train.
  svmType :: C.SVMType
, -- | The type of kernel function to use.
  kernelType :: C.KernelFunction
, -- | The degree @d@ in 'polynomial' kernel functions.
  degree :: Int
, -- | The value of @gamma@ for 'polynomial', 'rbf' or 'sigmoid' kernel
  -- functions.
  gamma :: Double
, -- | The added constant @c0@ in the 'polynomial' and 'sigmoid'
  -- kernel functions
  c0  :: Double
, -- | The cache size to use during training specified in MB.
  cacheSize :: Double
, -- | The stopping criterion.
  epsilon :: Double
, -- | The C paramter for the SVM types 'cSVC', 'epsilonSVR' and 'nuSVR'.
  parameterC :: Double
, -- | This list can be used to change the penalty for some classes. If the 
  -- weight for a class is not changed, it is set to 1.
  -- This is useful for training classifier using unbalanced input data or 
  -- with asymmetric misclassification cost.
  -- If you don't want to use the feature provided by this and the next 
  -- field, just set this field to the empty list @[]@.
  -- This field contains a list of @(classLabel, weight)@ pairs.
  labelWeights :: [(Int, Double)]
, -- | The parameter @nu@ used for the SVM types @'nuSVC'@, @'nuSVR'@, and 
  -- @'oneClass'@. It approximates the fraction of training errors and 
  -- support vectors.
  nu :: Double
, -- | This is the epsilon in epsilon-insensitive loss function of 
  -- epsilon-SVM regression, i.e. SVM type @'epsilonSVR'@.
  epsilon' :: Double
, -- | This flag decides whether to use the shrinking heuristics or not.
  -- If set to 'True' shrinking is turned on, otherwise it's turned off.
  shrinking :: Bool
, -- | This flag decides whether to obtain a model with probability 
  -- information or not. Probability information is gathered if and only 
  -- if this flag is set to 'True'.
  probability :: Bool
} deriving Show

-- | A set of default parameters for nu-SVM classification.
defaultNu :: Parameters
defaultNu = Parameters nuSVC rbf 0 1 0 100 0.00001 0 [] 0.1 0 True False

-- | A handle to a model used by LibSVM. It can only be created by 
-- functions returning this type and those functions ensure proper
-- memory management
newtype Model = Model (ForeignPtr C.Model)

--------------------------------------------------------------------------------
-- Convenience Functions
--------------------------------------------------------------------------------

-- | Translates a @TrainingInput@ into a the problem format suitable as 
-- input for LibSVM's 'C.train' function.
handover :: TrainingInput -> IO C.Problem
handover input = let count = fromIntegral $ length input in do
  labels <- newArray $ map (realToFrac . getLabel) input
  ivs <- mapM (toSparse . getVector) input >>= newArray
  return $! C.Problem { C.size = count, C.labels = labels, C.inputs = ivs}

-- | Translates an @InputVector@ into the sparse representation expected 
-- by LibSVM.
toSparse :: InputVector -> IO C.NodeP
toSparse (InputVector v) = let 
  result =  reverse $ (-1, 0) : snd (foldl' f (0, []) v)
  f (c, xs) x = (c + 1, if x == 0 then xs else (c + 1, x) : xs) 
  node (index, value) = C.Node (fromIntegral index) (realToFrac value) in
  newArray $! map node $ result

-- | Translates the type @'Parameters'@ into a value of type 
-- @'ForeignPtr C.Parameters'@ by converting internal dataypes to the C 
-- datatype representations,  allocating the needed arrays and then 
-- associating the result with a finalizer.
marshallParameters :: Parameters -> IO (ForeignPtr C.Parameters)
marshallParameters Parameters {
  svmType       = t, 
  kernelType    = k, 
  degree        = d, 
  gamma         = g, 
  c0            = c0,
  cacheSize     = cs,
  epsilon       = e,
  parameterC    = c,
  labelWeights  = lws,
  nu            = nu,
  epsilon'      = e',
  shrinking     = sh,
  probability   = ps} = let 
    fI = fromIntegral
    rtf = realToFrac 
    weightCount = fI . length $ lws
    labels = map (fI . fst) lws
    weights = map (rtf . snd) lws in do
  labelArray <- newArray labels
  weightArray <- newArray weights
  new C.Parameters {
    C.svm_type      = t,
    C.kernel_type   = k,
    C.degree        = fI d,
    C.gamma         = rtf g,
    C.c0            = rtf c0,
    C.cache_size    = rtf cs,
    C.epsilon       = rtf e,
    C.parameterC    = rtf c,
    C.weight_labels = weightCount,
    C.weight_label  = labelArray,
    C.weight        = weightArray,
    C.nu            = rtf nu,
    C.epsilon'      = rtf e',
    C.shrinking     = fromBool sh,
    C.probability   = fromBool ps
  } >>= newForeignPtr C.finalizeParameters

-------------------------------------------------------------------------------
-- SVM Functions
-------------------------------------------------------------------------------

-- | Constructs and returns an SVM model according to
-- the given training data and parameters.
-- Throws a @'userError'@ if the @'Parameters'@ are deemed infeasible.
train :: Trainable i => i -> Parameters -> IO Model
train i parameters = let input = trainingInput i in do
  problem <- handover input >>= new
  c_parameters <- marshallParameters parameters
  model <- withForeignPtr c_parameters $ \p -> do
    check <- C.check_parameters problem p 
    if check == nullPtr 
      then C.train problem p 
      else do
        error_code <- peekCAString check
        throwIO $ userError 
          ("in train: check_parameters returned '" ++ error_code ++ "'")
  result <- newForeignPtr C.finalizeModel model
  return $! Model result

-- | Saves a model to a file.
-- Throws a @'userError'@ if an error occurs.
save :: Model -> FilePath -> IO ()
save (Model modelPointer) destination = 
  withCString destination $ \p -> throwIfNeg_
    (\code -> "in save: saving model to '" ++ destination ++ 
      "'returned failed with " ++ show code)
    (withForeignPtr modelPointer $ (C.save_model p))

-- | Loads a model from a file.
-- Throws a @'userError'@ if loading fails.
load :: FilePath -> IO Model
load source = withCString source $ \p -> throwIfNull
    ("in load: loading model from file '" ++ source ++ "' failed")
    (C.load_model p) >>= 
  newForeignPtr C.finalizeModel >>= return . Model
