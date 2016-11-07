# Autoencoder 
###[Author: Hussain Mir Ali]
An autoencoder neural network with single hidden layer and multiclass ouput. This project has been written in JavaScript. 

##External Librarbies Used:
* csv-parse License: https://github.com/wdavidw/node-csv-parse/blob/master/LICENSE
* mathjs License: https://github.com/josdejong/mathjs/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE

##Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

##Installation:
*  Download the project and unzip it.
*  Copy the 'autoencoder' folder to your node_modules folder in your project directory.
*  Require it using 'require('autoencoder')' in your main JavaScript file.
*  If you want to reinstall node_modules for this project then run 'sudo npm install -g" in your terminal under the 'autoencoder' project directory.

##Testing:
* For unit testing Mocha and Sinon have been used. 
* On newer computers run the command 'mocha --timeout 50000', the 50000 ms timeout is to give enough time for tests to complete as they might not process before timeout. 
* On older computers run the command 'mocha --timeout 300000', the 300000 ms timeout is to give enough time for tests to complete as they might not process before timeout on older computers. 
* If need be more than 300000 ms should be used to run the tests depending on the processing power of the computer. 

##Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.

###Sample usage:

```javascript
var Autoencoder = require('autoencoder');
var callback_data;

var callback = function (data) {
    console.log(data);
    callback_data = data;
};

var an = new Autoencoder({
        'path': path,
        /*optional path to save the weights.*/
        'hiddenLayerSize': 6,
        'p': 0.05,/*Sparsity parameter.*/
        'beta': 0.3,/*Weight of the sparsity term.*/
        'learningRate': 0.9,
        'algorithm_mode': 0 /*This is to specify if  testing:2, cross validating:1 or training:0 data.*/ ,
        'threshold_value': undefined /*optional threshold value*/ ,
        'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting.*/ ,
        'optimization_mode': {
          'mode': 0
        } /*optional optimization mode for type of gradient descent.*/ ,
        'notify_count': 10 /*optional value to execute the callback after every x number of iterations.*/ ,
        'iteration_callback': callback /*optional callback that can be used for getting cost and iteration value on every notify count.*/ ,
        'maximum_iterations': 500 /*optional maximum iterations to be allowed.*/
      });

an.train_network([
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

*/
```
