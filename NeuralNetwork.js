/**
Copyright (c) 2015-2016 Hussain Mir Ali
**/
"use strict";

/**
 * The NeuralNetwork class contains all the necessary logic to train data for multiclass classification using single layer Neural Network.
 *
 * @class NeuralNetwork
 * @constructor
 * @param {Object} args Contains all the necessary parameters for the neural network as listed below.
 * @param {String} args.path optional Path to save the weights.
 * @param {Number} args.learningRate Learning rate for BackPropogation.
 * @param {Number} args.threshold_value Optional threshold value for error. 
 * @param {Number} args.regularization_parameter Optional regularization parameter to prevent overfitting. 
 * @param {Number} args.notify_count Optional value to execute the iteration_callback after every x number of iterations.
 * @param {Function} args.iteration_callback Optional callback that can be used for getting cost and iteration value on every notify count.
 * @param {Number} args.maximum_iterations Optional maximum iterations to be allowed before the optimization is complete. 
 **/

var NeuralNetwork = function(args) {
    this.fs = require('fs');
    this.parse = require('csv-parse');
    this.MathJS = require('mathjs');
    this.q = require('q');
    this.initArgs = args;
    this.path = args.path || new Array('./data/Weights_Layer1.txt', './data/Weights_Layer2.txt');
    this.threshold = args.threshold || (1 / this.MathJS.exp(6));
    this.algorithm_mode = 0;
    this.iteration_callback = args.iteration_callback;
    this.regularization_param = args.regularization_param || 0.01;
    this.learningRate = args.learningRate || 0.5;
    this.maximum_iterations = args.maximum_iterations || 1000;
    this.notify_count = args.notify_count || 100;
};

/**
 * This method returns all the parameters passed to the constructor.
 *
 * @method getInitParams
 * @return {Object} Returns the constructor parameters.
 */

NeuralNetwork.prototype.getInitParams = function() {
    return {
        'algorithm_mode': this.algorithm_mode,
        'path': this.path,
        'notify_count': this.notify_count,
        'iteration_callback': this.iteration_callback,
        'threshold': this.threshold,
        'regularization_param': this.regularization_param,
        'learningRate': this.learningRate,
        'maximum_iterations': this.maximum_iterations
    };
};

/**
 * This method serves as the logic for the sigmoid function.
 *
 * @method sigmoid
 * @param {matrix} z The matrix to be used as the input for the sigmoid function. 
 * @return {matrix} Returns the elementwise sigmoid of the input matrix.
 */

NeuralNetwork.prototype.sigmoid = function(z) {
    var scope = {
            z: (typeof(z) === "number") ? this.MathJS.matrix([
                [z]
            ]) : z
        },
        sigmoid;

    scope.ones = this.MathJS.ones(scope.z.size()[0], scope.z.size()[1]);
    sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))
    return sigmoid;
};

/**
 *This method is responsible for the forwardPropagation in the Neural Network.
 *
 * @method forwardPropagation 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} W1 The matrix representing the weights for layer 1.
 * @param {matrix} W2 The matrix representing the weights for layer 2.
 * @return {matrix} Returns the resultant ouput of forwardPropagation.
 */
NeuralNetwork.prototype.forwardPropagation = function(X, W1, W2) {
    var y_result, X = this.MathJS.matrix(X) || this.x;
    this.W1 = W1 || this.W1;
    this.W2 = W2 || this.W2;
    this.z2 = this.MathJS.multiply(X, this.W1);
    this.a2 = this.sigmoid(this.z2);
    this.z3 = this.MathJS.multiply(this.a2, this.W2);
    y_result = this.sigmoid(this.z3);
    return y_result;
};

/**
 * This method serves as the logic for the sigmoid function derivative.
 *
 * @method sigmoid_Derivative
 * @param {matrix} z The matrix to be used as the input for the sigmoid function derivative. 
 * @return {matrix} Returns the elementwise sigmoid derivative of the input matrix.
 */
NeuralNetwork.prototype.sigmoid_Derivative = function(z) {
    var scope = {
            z: z
        },
        sigmoid_Derivative;
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    sigmoid_Derivative = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

    return sigmoid_Derivative;
};

/**
 *This method is responsible for the costFunction, i.e. error.
 *
 * @method costFunction 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} Y The output matrix corresponding to training data.
 * @param {Number} algorithm_mode The current algorithm mode (testing: 2, crossvalidating: 1, training: 0).
 * @return {Number} Returns the resultant cost.
 */
NeuralNetwork.prototype.costFunction = function(X, Y, algorithm_mode) {
    var J;
    var scope = {};
    this.y_result = this.forwardPropagation(X || this.x, undefined, undefined);
    scope.y_result = this.y_result;
    scope.y = this.MathJS.matrix(Y);
    scope.x = this.MathJS.matrix(X);
    scope.W1 = this.W1;
    scope.W2 = this.W2;
    this.algorithm_mode = algorithm_mode || this.algorithm_mode;

    if (this.algorithm_mode === 0)
        J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]) + (this.regularization_param / 2) * (this.MathJS.sum(this.MathJS.eval('W1.^2', scope)) + this.MathJS.sum(this.MathJS.eval('W2.^2', scope))); //regularization parameter
    else if (this.algorithm_mode === 1 || this.algorithm_mode === 2)
        J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]);

    return J;
};

/**
 *This method is responsible for the costFunction_Derivative, i.e. gradient of error with respect to weights.
 *
 * @method costFunction_Derivative 
 * @param {matrix} X The input matrix representing the features.
 * @param {matrix} Y The output matrix corresponding to training data.
 * @param {matrix} W1 The matrix representing the weights for layer 1.
 * @param {matrix} W2 The matrix representing the weights for layer 2.
 * @return {Array} Returns the resultant gradients with respect to layer 1: dJdW1 and layer 2: dJdW2 of the Neural Network.
 */

NeuralNetwork.prototype.costFunction_Derivative = function(X, Y, W1, W2) {
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

    scope.arrA = this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2));
    scope.arrB = this.sigmoid_Derivative(this.z2);

    var del_2 = this.MathJS.eval('arrA.*arrB', scope);
    var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(scope.x), del_2);

    scope.dJdW1 = dJdW1;
    scope.regularization_term_dJdW1 = this.MathJS.eval('W1.*regularization_param', scope);
    dJdW1 = this.MathJS.eval('dJdW1.*(1/m) + regularization_term_dJdW1', scope);

    return [dJdW1, dJdW2];

};

/**
 *This method is responsible for saving the trained weights of the Neural Network to text files at specified path.
 *
 * @method saveWeights 
 * @param {Array} weights The weights of the layer1 and layer2 of the Neural Network.
 * @param {String} path The path at wich the weights are to be saved.
 * @return {Boolean} Returns true after succesfuly saving the weights.
 */
NeuralNetwork.prototype.saveWeights = function(weights, path) {
    var defered = this.q.defer();

    try {
        this.fs.writeFileSync(path[0], weights[0].toString().replace(/\]\,\s\[/g, "\n").replace("[[", "").replace("]]", ""));
        this.fs.writeFileSync(path[1], weights[1].toString().replace(/\]\,\s\[/g, "\n").replace("[[", "").replace("]]", ""));
    } catch (e) {
        return false;
    }

    console.log("\nWeights were successfuly saved.");
    return true;
};

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
NeuralNetwork.prototype.gradientDescent = function(X, Y, W1, W2) {
    var gradient = new Array(2),
        self = this,
        x = X || this.x,
        y = Y || this.y,
        W1 = W1,
        W2 = W2,
        cost,
        scope = {},
        defered = this.q.defer(),
        path = this.path,
        i = 0;

    if (this.algorithm_mode == 0)
        console.log('Training ...\n');

    while (1) {
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
            cost = this.costFunction(x, y);
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
            this.saveWeights([this.W1, this.W2], path);
            defered.resolve([cost, i]);
            break;
        }
    }

    return defered.promise;
};

/**
 *This method is responsible for creating layers and initializing random weights. 
 *
 * @method train_network
 * @param {matrix} X The input matrix representing the features of the training set.
 * @param {matrix} Y The output matrix corresponding to training set data.
 * @return {Object} Returns a resolved promise with iteration and cost data on successful completion of optimization. 
 */
NeuralNetwork.prototype.train_network = function(X, Y) {
    this.x = this.MathJS.matrix(X);
    this.y = this.MathJS.matrix(Y);
    this.algorithm_mode = 0;

    if ((this.y.size()[0] !== this.x.size()[0])) {
        console.log('\nPlease change the size of the input matrices so that X and Y have same number of rows.');
    } else {
        this.inputLayerSize = this.x.size()[1];
        this.outputLayerSize = 1;
        this.hiddenLayerSize = this.x.size()[1] + 1;
        this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), -10, 10));
        this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize * this.y.size()[1]]), -10, 10));
    }
    return this.gradientDescent(undefined, undefined, undefined, undefined);
};

/**
 *This contains logic to predict result for a given input after training on data. 
 *
 * @method predict_result
 * @param {matrix} X The input matrix representing the features.
 * @return {matrix} Returns the resultant matrix after performing forwardPropagation on saved weights.
 */

NeuralNetwork.prototype.predict_result = function(X) {
    var y_result;
    this.setWeights(this.path);
    y_result = this.forwardPropagation(X);
    return y_result;
};

/**
 *This method is responsible for setting weights of the Neural Network from a specified path.
 *
 * @method setWeights 
 * @param {String} path The path where the trained weights are to be found.
 * @return {Object} Returns a resolved promise after successfuly setting weights.
 */
NeuralNetwork.prototype.setWeights = function(path) {
    var contents_layer1, contents_layer2;
    var dataA = this.fs.readFileSync(path[0], 'utf8');
    var dataB = this.fs.readFileSync(path[1], 'utf8');
    var self = this,
        success;

    return (function() {
        var defered = self.q.defer();
        self.parse(dataA, {}, function(err, array) {
            if (err) {
                throw (err);
            }
            defered.resolve();
            contents_layer1 = self.MathJS.matrix(array);
        })

        return defered.promise;
    })().then(
        function() {

            var defered = self.q.defer();

            self.parse(dataB, {}, function(err, array) {
                if (err) {
                    throw (err);
                }
                contents_layer2 = self.MathJS.matrix(array);
                if (contents_layer1 !== undefined && contents_layer2 !== undefined) {
                    self.W1 = (contents_layer1);
                    self.W2 = (contents_layer2);
                    defered.resolve({
                        'success': true
                    });
                } else {
                    defered.reject({
                        'success': false
                    });
                }
            });

            return defered.promise;

        });
};

/**
 *This method is responsible for producing the cross validation error after training data.
 *
 * @method cross_validate_network
 * @param {matrix} X The input matrix representing the features of the cross validation set.
 * @param {matrix} Y The output matrix corresponding to training data of the cross validation set.
 * @return {Number} Returns an error value associated with the cross validation.
 */
NeuralNetwork.prototype.cross_validate_network = function(X, Y) {
    console.log("\n Cross Validating...");
    this.algorithm_mode = 1;
    return this.costFunction(X, Y, undefined);
};

/**
 *This method is responsible for producing the test error after training data.
 *
 * @method test_network
 * @param {matrix} X The input matrix representing the features of the test set.
 * @param {matrix} Y The output matrix corresponding to training data of the test set.
 * @return {Number} Returns an error value associated with testing.
 */
NeuralNetwork.prototype.test_network = function(X, Y) {
    console.log("\n Testing...");
    this.algorithm_mode = 2;
    return this.costFunction(X, Y, undefined);
};

module.exports = NeuralNetwork;
