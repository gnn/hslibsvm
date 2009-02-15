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
  -- ** Training
, accuracy
, crossvalidate
, train   
  -- ** Serialization
, load
, save
  -- ** Model Querying 

, labels
, countClasses 
, trainedType
, svrProbability

  -- ** Prediction
, decisionValues
, predict
, probabilities

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

-- | Throws a @'userError'@ constructed with the supplied string, if 
-- the supplied model doesn't contain probability information.
checkProbabilities :: Model -> String -> IO ()
checkProbabilities m@(Model mfp) location = throwIf_ 
  (not . toBool) 
  (\_ -> "in " ++ location ++ 
    ": supplied model doesn't contain probability information")
  (withForeignPtr mfp C.check_probability_model)


-- | A list of SVM types which aren't classification SVMs.
nonClassifiers :: [C.SVMType]
nonClassifiers = [oneClass, epsilonSVR, nuSVR]

-- | Translates instances of class @'Trainable'@ into the problem format 
-- suitable as input for LibSVM's 'C.train' function.
marshalInput :: Trainable i => i -> IO C.Problem
marshalInput input = let 
  ti = trainingInput input
  labels = map (realToFrac . getLabel) ti
  count = fromIntegral $ length ti in do
  labelArray <- newArray labels
  vectorArray <- mapM (newArray . toSparse) ti >>= newArray
  return $! 
    C.Problem { C.size = count, C.labels = labelArray, C.inputs = vectorArray}

-- | Translates an @InputVector@ into the sparse representation expected 
-- by LibSVM.
toSparse :: SVMInput input => input -> [C.Node]
toSparse input = map node result where
  result =  reverse $ (-1, 0) : snd (foldl' f (0, []) v)
  (InputVector v) = inputVector input
  f (c, xs) x = (c + 1, if x == 0 then xs else (c + 1, x) : xs) 
  node (index, value) = C.Node (fromIntegral index) (realToFrac value)

-- | Translates the type @'Parameters'@ into a value of type 
-- @'ForeignPtr C.Parameters'@ by converting internal dataypes to the C 
-- datatype representations,  allocating the needed arrays and then 
-- associating the result with a finalizer.
marshalParameters :: Parameters -> IO (ForeignPtr C.Parameters)
marshalParameters Parameters {
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
train input parameters = do
  problem <- marshalInput input >>= new
  c_parameters <- marshalParameters parameters
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

-- | Classifies the input.
-- @predict model x@ does classification on the input vector @x@ 
-- according to @model@.
--
-- For a classification model, the predicted class for x is returned. 
-- For a regression model, the function value of x calculated using 
-- the model is returned. For an one-class model, +1 or -1 is returned. 
predict :: SVMInput input => Model -> input -> IO Label
predict (Model modelFP) input = withForeignPtr modelFP $ \modelP ->
  withArray (toSparse input) (C.predict modelP) >>= return . realToFrac

-- | Conducts cross validation. 
-- @crossvalidate input parameters n@ target separates input into @n@ folds. 
-- Using the given @parameters@, sequentially each fold is predicted using
-- the model from training with the remaining folds. 
-- Thus all the input vectors are predicted once and the list of predicted 
-- labels is returned.
crossvalidate :: Trainable i => i -> Parameters -> Int -> IO [Double]
crossvalidate i p n = let 
  size = length $ trainingInput i 
  cn = fromIntegral n in do
  problem <- marshalInput i
  parameters <- marshalParameters p
  with problem $ \problemP -> 
    withForeignPtr parameters $ \parametersP -> 
      allocaArray size $ \buffer ->
        C.cross_validation problemP parametersP cn buffer >>
        peekArray size buffer >>= return . map realToFrac

-- | Computes the accuracy gained by training with the given @parameters@.
-- @accuracy input parameters n@ calls @'crossvalidate'@ with the given
-- arguments and computes the accuracy from the result. The accuracy is the
-- percentage of the labels predicted correctly.
accuracy :: Trainable i => i -> Parameters -> Int -> IO Double
accuracy i p n = let 
  labels = map getLabel $ trainingInput i
  maximum = fromIntegral $ length labels in do
  predicted <- crossvalidate i p n
  let hits = fromIntegral $ length $ filter (id) $ zipWith (==) labels predicted
  return $! hits * 100 / maximum

-- | @'countClasses model'@ returns the number of classes of the 
-- classification model @model@. Returns 2 if @model@ is a regression
-- or a one-class model.
countClasses :: Model -> IO Int
countClasses (Model m) = 
  withForeignPtr m C.get_nr_class >>= return . fromIntegral

-- | Returns the @'C.SVMType'@ of the model.
trainedType :: Model -> IO C.SVMType
trainedType (Model m) = withForeignPtr m C.get_svm_type

-- | @'labels model'@ returns a list with the labels present in the given
-- @model@. If @model@ is a one-class or a regression model then the 
-- empty list is returned.
labels :: Model -> IO [Int]
labels m@(Model mfp) = do
  t <- trainedType m
  if t `elem` nonClassifiers 
    then return [] 
    else do 
      classes <- countClasses m
      withForeignPtr mfp $ \mp -> allocaArray classes $ \ip -> 
        C.get_labels mp ip >> 
        peekArray classes ip >>= 
        return . map fromIntegral

-- | For a regression model with probability information, this function
-- outputs a value sigma > 0. For test data, we consider the probability
-- model: 
--
-- * @target value = predicted value + z, 
-- z: Laplace distribution e^(-|z|sigma)(2sigma)@
--
-- NOTE: This is copied pretty much verbatim from the LibSVM README and 
-- I don't really have a clue what it means. If anybody has a good 
-- explanation, an email would be greatly appreciated and would be used
-- to clarify this.
--
-- Throws a @'userError'@ if the model doesn't contain probability 
-- information.
svrProbability :: Model -> IO Double
svrProbability m@(Model mfp) = withForeignPtr mfp $ \mp ->
  checkProbabilities m "svrProbability" >>
  C.get_svr_probability mp >>= return . realToFrac

-- | Returns decision values for a given test vector and model.
-- 
-- For a classification model @decisionValues model x@ 
-- will return a function which accepts two labels @l1@ and @l2@ as its 
-- parameters and will return the corresponding decision value for @x@ 
-- when considering the two class SVM @l1@ vs. @l2@. 
-- The possible label values can be obtained via @'labels'@.
--
-- For a regression model, the returned funcion will be a constant 
-- function always returning the function value of x calculated using 
-- the model while for a one-class model it will be the constant function
-- returning +1 or -1. 
decisionValues :: SVMInput i => i -> Model -> (Int -> Int -> IO Double)
decisionValues i m@(Model mfp) l1 l2 = let 
  f a b = if a < b then (a, b) else (b, a)
  key = f l1 l2
  in do
  t <- trainedType m
  if t `elem` nonClassifiers 
    then predict m i
    else do 
      ls <- labels m 
      let l = length ls
      let n = l * (l - 1) `div` 2
      results <- mallocForeignPtrArray n
      withForeignPtr results $ \rp -> withForeignPtr mfp $ \mp -> do
        withArray (toSparse i) (\a -> C.predict_values mp a rp)
        dvs <- peekArray n rp >>= return . map realToFrac
        let 
          table = zip [f (ls !! i) (ls !! j) | i<-[0..l-2], j<-[i+1..l-1]] dvs
        return $! 
          maybe 
            (error $ "while looking up decision values: " ++
              "illegal key '" ++ show key ++ "'")
            id
            (lookup key table)

-- | If @model@ is a classification model with probability information,
-- @probabilities model x@ returns a pair @(p, ps)@ where @ps@ is a list
-- of probabilities such that @(ps !! i)@ is the probability of @x@
-- beeing labeled with @('labels' model !! i)@ and @p@ is the label with
-- the maximum probability.
-- If @model@ belongs to a regression/one-class SVM, @ps = [p]@ and @p@
-- will be the result of @'predict' model x@.
-- Throws a @'userError'@ if @model@ is a classification model but 
-- contains no probability information.
probabilities :: SVMInput i => Model -> i -> IO (Double, [Double])
probabilities m@(Model mfp) i = do
  t <- trainedType m
  if t `elem` nonClassifiers 
    then do {p <- predict m i; return $! (p, [p])}
    else do
      checkProbabilities m "probabilities"
      ls <- labels m
      let l = length ls
      allocaArray l $ \dp -> withForeignPtr mfp $ \mp -> do
        withArray (toSparse i) $ \ip -> do
          p <- C.predict_probability mp ip dp
          ps <- peekArray l dp
          return $! (realToFrac p, map realToFrac ps)

