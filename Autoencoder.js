/**
Copyright (c) 2016-2017 Hussain Mir Ali
**/
"use strict";

var window_object = (function(g){
      return g;
  }(this));

/**
 * The Autoencoder class contains all the necessary logic to train data for multiclass classification using single layer Autoencoder.
 *
 * @class Autoencoder
 * @constructor
 * @param {Object} args Contains all the necessary parameters for the Autoencoder as listed below.
 * @param {Number} args.learningRate Learning rate for BackPropogation.
 * @param {Number} args.threshold_value Optional threshold value for error. 
 * @param {Number} args.p Sparsity parameter for the autoencoder. 
 * @param {Number} args.beta Weight of the sparsity term. 
 * @param {Number} args.regularization_parameter Optional regularization parameter to prevent overfitting. 
 * @param {Number} args.notify_count Optional value to execute the iteration_callback after every x number of iterations.
 * @param {Function} args.iteration_callback Optional callback that can be used for getting cost and iteration value on every notify count.
 * @param {Number} args.hiddenLayerSize Optional value for number of hidden layer units.
 * @param {Number} args.maximum_iterations Optional maximum iterations to be allowed before the optimization is complete. 
 * @param {Object} args.optimization_mode Optional optimzation(gradient descent) mode. For batch gradient descent optimization_mode =  {'mode': 0} and for mini-batch gradient descent optimization_mode =  {'mode': 1, 'batch_size': (specify  required size) }. 
 **/

class Autoencoder {

  constructor(args){
   if(Object.keys(window_object).length === 0){
        this.MathJS = require('mathjs');
        this.q = require('q');
    }
    else{
        this.MathJS = math;
        this.q = Q;
    }

  this.initArgs = args;
  this.threshold = args.threshold || (1 / this.MathJS.exp(3));
  this.algorithm_mode = 0;
  this.iteration_callback = args.iteration_callback;
  this.hiddenLayerSize = args.hiddenLayerSize;
  this.regularization_param = args.regularization_param || 0.01;
  this.learningRate = args.learningRate || 0.5;
  this.p_prime = [];
  this.p = args.p;
  this.beta = args.beta;
  this.optimization_mode = (args.optimization_mode === undefined) ? {
    'mode': 0
  } : args.optimization_mode;
  this.maximum_iterations = args.maximum_iterations || 1000;
  this.batch_iteration_count = 0;
  this.notify_count = args.notify_count || 100;
}

/**
 * This method returns all the parameters passed to the constructor.
 *
 * @method getInitParams
 * @return {Object} Returns the constructor parameters.
 */

getInitParams() {
  return {
    'algorithm_mode': this.algorithm_mode,
    'beta': this.beta,
    'p': this.p,
    'hiddenLayerSize': this.hiddenLayerSize,
    'optimization_mode': this.optimization_mode,
    'notify_count': this.notify_count,
    'iteration_callback': this.iteration_callback,
    'threshold': this.threshold,
    'regularization_param': this.regularization_param,
    'learningRate': this.learningRate,
    'maximum_iterations': this.maximum_iterations
  };
}

/**
 * This method serves as the logic for the sigmoid function.
 *
 * @method sigmoid
 * @param {matrix} z The matrix to be used as the input for the sigmoid function. 
 * @return {matrix} Returns the elementwise sigmoid of the input matrix.
 */

sigmoid(z) {
  var scope = {
      z: (typeof(z) === "number") ? this.MathJS.matrix([
        [z]
      ]) : z
    },
    sigmoid;
  if (scope.z.size()[1] === undefined)
    scope.ones = this.MathJS.ones(scope.z.size()[0]);
  else
    scope.ones = this.MathJS.ones(scope.z.size()[0], scope.z.size()[1]);
  sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))
  return sigmoid;
}
/**
 *This method is responsible for the forwardPropagation in the Autoencoder.
 *
 * @method forwardPropagation 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} W1 The matrix representing the weights for layer 1.
 * @param {matrix} W2 The matrix representing the weights for layer 2.
 * @return {matrix} Returns the resultant ouput of forwardPropagation.
 */
forwardPropagation(X, W1, W2) {
  var y_result, X = this.MathJS.matrix(X) || this.x, scope={};
  this.W1 = W1 || this.W1;
  this.W2 = W2 || this.W2;
  this.z2 = this.MathJS.multiply(X, this.W1);
  this.a2 = this.sigmoid(this.z2);
  this.p_prime = this.MathJS.mean(this.a2,0);
  this.z3 = this.MathJS.multiply(this.a2, this.W2);
  y_result = this.sigmoid(this.z3);
  return y_result;
}

/**
 *This method is responsible for calculating the Kullback-Leibler sum in the Autoencoder.
 *
 * @method kullback_leibler_sum
 * @return {Number} sum Returns the Kullback-Leibler sum.
 */
kullback_leibler_sum() {
  var scope={}, sum, kl_matrix, sum;
  scope.ones = this.MathJS.ones([this.a2.size()[1]]);
  scope.p = this.p;
  scope.p = this.MathJS.eval('ones.*p',scope);
  scope.p_prime = this.p_prime;

  kl_matrix = this.MathJS.eval('(log(p./p_prime).*p)+(log((1-p)./(1-p_prime)).*(1-p))', scope);
  sum = this.MathJS.sum(kl_matrix);

  return sum;
}

/**
 * This method serves as the logic for the sigmoid function derivative.
 *
 * @method sigmoid_Derivative
 * @param {matrix} z The matrix to be used as the input for the sigmoid function derivative. 
 * @return {matrix} Returns the elementwise sigmoid derivative of the input matrix.
 */
sigmoid_Derivative(z) {
  var scope = {
      z: z
    },
    sigmoid_Derivative;
  scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
  sigmoid_Derivative = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

  return sigmoid_Derivative;
}
/**
 *This method is responsible for the costFunction, i.e. error.
 *
 * @method costFunction 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} Y The output matrix corresponding to training data.
 * @param {Number} algorithm_mode The current algorithm mode (testing: 2, crossvalidating: 1, training: 0).
 * @return {Number} Returns the resultant cost.
 */
costFunction(X, Y, algorithm_mode) {
  var J, batch_size;
  var scope = {};
  var iteration_count;

  scope.y = this.MathJS.matrix(Y);
  scope.x = this.MathJS.matrix(X);
  scope.W1 = this.W1;
  scope.W2 = this.W2;
  this.algorithm_mode = algorithm_mode || this.algorithm_mode;

  this.batch_iteration_count++;
  iteration_count = this.batch_iteration_count;

  if (this.optimization_mode.mode === 1) { //cost for mini-batch gradient descent.
    batch_size = this.optimization_mode.batch_size;

    scope.x = this.MathJS.matrix(scope.x._data.slice(batch_size * iteration_count, batch_size + (batch_size * iteration_count)));
    scope.y = this.MathJS.matrix(scope.y._data.slice(batch_size * iteration_count, batch_size + (batch_size * iteration_count)));

    if (iteration_count * this.optimization_mode.batch_size === scope.x.size()[0]) {
      this.batch_iteration_count = 0;
    }

    this.y_result = this.forwardPropagation(scope.x, undefined, undefined);
    scope.y_result = this.y_result;

    if (this.algorithm_mode === 0)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (this.optimization_mode.batch_size) + (this.regularization_param / 2) * (this.MathJS.sum(this.MathJS.eval('W1.^2', scope)) + this.MathJS.sum(this.MathJS.eval('W2.^2', scope))); //cost with regularization parameter.
    else if (this.algorithm_mode === 1 || this.algorithm_mode === 2)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (this.optimization_mode.batch_size);
  } else if (this.optimization_mode.mode === 2 && (iteration_count <= scope.x.size()[0])) { // cost for stochastic gradient descent.
    scope.x = this.MathJS.matrix(scope.x._data[iteration_count - 1]);
    scope.y = this.MathJS.matrix(scope.y._data[iteration_count - 1]);

    this.y_result = this.forwardPropagation(scope.x, undefined, undefined);
    scope.y_result = this.y_result;

    if (this.algorithm_mode === 0)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) + (this.regularization_param / 2) * (this.MathJS.sum(this.MathJS.eval('W1.^2', scope)) + this.MathJS.sum(this.MathJS.eval('W2.^2', scope))); //cost with regularization parameter.
    else if (this.algorithm_mode === 1 || this.algorithm_mode === 2)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope));
  } else if (this.optimization_mode.mode === 0) { //cost with batch gradient descent.
    this.y_result = this.forwardPropagation(scope.x, undefined, undefined);
    scope.y_result = this.y_result;

    if (this.algorithm_mode === 0)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]) + (this.regularization_param / 2) * (this.MathJS.sum(this.MathJS.eval('W1.^2', scope)) + this.MathJS.sum(this.MathJS.eval('W2.^2', scope))); //cost with regularization parameter.
    else if (this.algorithm_mode === 1 || this.algorithm_mode === 2)
      J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]);

  }

  return J + this.beta*this.kullback_leibler_sum();
}

/**
 *This method is responsible for the costFunction_Derivative, i.e. gradient of error with respect to weights.
 *
 * @method costFunction_Derivative 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} Y The output matrix corresponding to training data.
 * @param {matrix} W1 The matrix representing the weights for layer 1.
 * @param {matrix} W2 The matrix representing the weights for layer 2.
 * @return {Array} Returns the resultant gradients with respect to layer 1: dJdW1 and layer 2: dJdW2 of the Autoencoder.
 */

costFunction_Derivative(X, Y, W1, W2) {
  if (this.optimization_mode.mode === 1) {
    var iteration_count = this.batch_iteration_count;
    var batch_size = this.optimization_mode.batch_size;
    var X = this.MathJS.matrix(this.x._data.slice(batch_size * iteration_count, batch_size + (batch_size * iteration_count)));
    var Y = this.MathJS.matrix(this.y._data.slice(batch_size * iteration_count, batch_size + (batch_size * iteration_count)));

  }

  this.y_result = this.forwardPropagation(X || this.x, undefined, undefined);
  var scope = {};
  scope.y_result = this.y_result;
  scope.y = Y || this.y;
  scope.x = X || this.x;
  scope.diff = this.MathJS.eval('-(y-y_result)', scope);
  scope.sigmoid_Derivative_z3 = this.sigmoid_Derivative(this.z3);
  scope.regularization_param = this.regularization_param;
  scope.W2 = W2 || this.W2;
  scope.W1 = W1 || this.W1;
  scope.m = scope.x.size()[0];

  var del_3 = this.MathJS.eval('diff.*sigmoid_Derivative_z3', scope);
  var dJdW2 = this.MathJS.multiply(this.MathJS.transpose(this.a2), del_3);
  scope.dJdW2 = dJdW2;
  scope.regularization_term_dJdW2 = this.MathJS.eval('W2.*regularization_param', scope);
  dJdW2 = this.MathJS.eval('dJdW2.*(1/m) + regularization_term_dJdW2', scope);
 
  scope.beta = this.beta;
  scope.p_prime = this.p_prime;
  scope.ones = this.MathJS.ones([this.a2.size()[1]]);
  scope.p = this.p;
  scope.p = this.MathJS.eval('ones.*p',scope);

  scope.arrA = this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2));
  scope.kl_term = this.MathJS.eval('(((p./p_prime).*-1)+((1-p)./(1-p_prime))).*beta',scope);
  scope.kl_matrix = [];

  for(var i=0; i<scope.arrA.size()[0]; i++){
    scope.kl_matrix.push(scope.kl_term);
  }

  scope.kl_matrix = this.MathJS.matrix(scope.kl_matrix);
  scope.arrA = this.MathJS.eval('arrA+kl_matrix',scope);
  scope.arrB = this.sigmoid_Derivative(this.z2);

  var del_2 = this.MathJS.eval('arrA.*arrB', scope);
  var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(scope.x), del_2);

  scope.dJdW1 = dJdW1;
  scope.regularization_term_dJdW1 = this.MathJS.eval('W1.*regularization_param', scope);
  dJdW1 = this.MathJS.eval('dJdW1.*(1/m) + regularization_term_dJdW1', scope);

  return [dJdW1, dJdW2];

}

/**
 *This method is responsible for saving the trained weights of the Autoencoder.
 *
 * @method saveWeights 
 * @param {Array} weights The weights of the layer1 and layer2 of the Autoencoder.
 * @return {Boolean} Returns true after succesfuly saving the weights.
 */
saveWeights(weights) {
  var defered = this.q.defer();
  if(Object.keys(window_object).length === 0){
    global.localStorage.setItem("Weights", weights);
  }
  else{
    localStorage.setItem("Weights", weights);
  }
  console.log("\nWeights were successfuly saved.");
  return true;
}

/**
 *This method is responsible for the optimization of weights, i.e. BackPropagation algorithm.
 *
 * @method gradientDescent
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} Y The output matrix corresponding to training data.
 * @param {matrix} W1 The matrix representing the weights for layer 1.
 * @param {matrix} W2 The matrix representing the weights for layer 2.
 * @return {Object} Returns a resolved promise with iteration and cost data on successful completion of optimization. 
 */
gradientDescent(X, Y, W1, W2) {
  var gradient = new Array(2),
    self = this,
    x = X || this.x,
    y = Y || this.y,
    W1 = W1,
    W2 = W2,
    cost,
    scope = {},
    defered = this.q.defer(),
    i = 1;

  if (this.algorithm_mode == 0)
    console.log('Training ...\n');

  while (true) {

    if (x !== undefined && y !== undefined && W1 !== undefined && W2 !== undefined)
      gradient = this.costFunction_Derivative(x, y, W1, W2);
    else
      gradient = this.costFunction_Derivative(undefined, undefined, undefined, undefined);
    scope.W1 = W1 || this.W1;
    scope.W2 = W2 || this.W2;
    scope.rate = this.learningRate;
    scope.dJdW1 = gradient[0];
    scope.dJdW2 = gradient[1];

    this.W2 = this.MathJS.eval('W2 - dJdW2.*rate', scope);
    this.W1 = this.MathJS.eval('W1 - dJdW1.*rate', scope);

    if (x !== undefined && y !== undefined)
      cost = this.costFunction(x, y, undefined);
    if (i % this.notify_count === 0 && this.iteration_callback !== undefined) {
      this.iteration_callback.apply(null, [{
        'cost': cost,
        'iteration': i /*iteration count*/ ,
        'Weights_Layer1': self.W1,
        'Weights_Layer2': self.W2
      }]); //notify cost values for diagnosing the performance of learning algorithm.
    }
    i++;
    if (i > this.maximum_iterations || cost <= (this.threshold)) {
      this.saveWeights([this.W1, this.W2]);
      defered.resolve([cost, i]);
      break;
    }
  }

  return defered.promise;
}

/**
 *This method is responsible for creating layers and initializing random weights. 
 *
 * @method train_network
 * @param {matrix} X The input matrix representing the features of the training set.
 * @param {matrix} Y The output matrix corresponding to training set data.
 * @return {Object} Returns a resolved promise with iteration and cost data on successful completion of optimization. 
 */
train_network(X, Y) {
  this.x = this.MathJS.matrix(X);
  this.y = this.MathJS.matrix(Y);
  this.algorithm_mode = 0;

  if ((this.y.size()[0] !== this.x.size()[0])) {
    console.log('\nPlease change the size of the input matrices so that X and Y have same number of rows.');
  } else {
    this.inputLayerSize = this.x.size()[1];
    this.outputLayerSize = 1;
    if (this.hiddenLayerSize !== undefined) {
      if (this.hiddenLayerSize >= this.x.size()[1] + 1)
        this.hiddenLayerSize = this.hiddenLayerSize;
      else
        this.hiddenLayerSize = this.x.size()[1] + 1;
    } else if (this.hiddenLayerSize === undefined)
      this.hiddenLayerSize = this.x.size()[1] + 1;

    this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), -10, 10));
    this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize * this.y.size()[1]]), -10, 10));
  }
  return this.gradientDescent(undefined, undefined, undefined, undefined);
}

/**
 *This contains logic to predict result for a given input after training on data. 
 *
 * @method predict_result
 * @param {matrix} X The input matrix representing the features.
 * @return {matrix} Returns the resultant matrix after performing forwardPropagation on saved weights.
 */

predict_result(X) {
  var y_result;
  this.setWeights();
  y_result = this.forwardPropagation(X);
  return y_result;
}

/**
 *This method is responsible for setting weights of the Autoencoder.
 *
 * @method setWeights 
 * @return {Object} Returns a resolved promise after successfuly setting weights.
 */
setWeights() {
  var self = this;
  var weights;
    if(Object.keys(window_object).length === 0){
       weights = global.localStorage.getItem("Weights");
    }else{
       weights = localStorage.getItem("Weights");
     }

     self.W1 = weights[0];
     self.W2 = weights[1];

     return [self.W1, self.W2];
}
}

if(Object.keys(window_object).length === 0){
    module.exports = Autoencoder;
}else{
    window['Autoencoder'] = Autoencoder;
}