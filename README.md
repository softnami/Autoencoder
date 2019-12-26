# Autoencoder 
### [Author: Hussain Mir Ali]

![Autoencoder](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png)

An autoencoder neural network with single hidden layer and multiclass ouput. This project has been written in JavaScript. 

## External Libraries Used:
* mathjs License: https://github.com/josdejong/mathjs/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE

## Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

## Installation:
*  Run 'npm i @softnami/autoencoder' .

### Sample usage:

```javascript
//main.js file
import {Autoencoder} from '@softnami/autoencoder';

const callback = function (data) {
    console.log(data);
};

const autoencoder = new Autoencoder({
    'hiddenLayerSize': 6,
    'p': 0.05,/*Sparsity parameter.*/
    'beta': 0.3,/*Weight of the sparsity term.*/
    'learningRate': 0.9,
    'threshold_value': undefined /* Optional threshold value for cost. Defaults to 1/(e^3). */,
    'regularization_parameter': 0.001 /*Optional regularization parameter to prevent overfitting. Defaults to 0.01.*/ ,
    'optimization_mode': {
      'mode': 0
    } /*Optional optimization mode for type of gradient descent. {mode:1, 'batch_size': <your size>} for mini-batch and {mode: 0} for batch. Defaults to batch gradient descent.*/ ,
    'notify_count': 10 /*Optional value to execute the iteration_callback after every x number of iterations. Defaults to 100.*/ ,
    'iteration_callback': callback /*Optional callback that can be used for getting cost and iteration value on every notify count. Defaults to empty function.*/ ,
    'maximum_iterations': 500 /*Optional maximum iterations to be allowed before the optimization is complete. Defaults to 1000.*/
  });

autoencoder.train_network([
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
], [
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
]).then(console.log("\nTraining done!\n"));  

```

## Testing:
* For unit testing Mocha and Sinon have been used. 
* Run 'npm test', if timeout occurs then increase timeout in test script.

## Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.
