# Autoencoder 
###[Author: Hussain Mir Ali]
An autoencoder neural network with single hidden layer and multiclass ouput. This project has been written in JavaScript. 

##External Libraries Used:
* mathjs License: https://github.com/josdejong/mathjs/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE
* q License: https://github.com/kriskowal/q/blob/v1/LICENSE

##Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

##Installation:
*  Download the project and unzip it.
*  Copy the 'autoencoder' folder to your project directory.

###Sample usage:

```javascript
//main.js file
var callback_data;

var callback = function (data) {
    console.log(data);
    callback_data = data;
};

var an = new window.Autoencoder({
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

```
```
<!--index.html-->
<!doctype html>
<html>
  <head>
  </head>
  <body >
        <script src="autoencoder/lib/q.js"></script>
        <script src="autoencoder/lib/math.js"></script>
        <script src="autoencoder/Autoencoder.js"></script>
	<!--Include the main.js file where you use the algorithm.-->
        <script src="main.js"></script>
</body>
</html>

*/
```

##Testing:
* For unit testing Mocha and Sinon have been used. 
* On newer computers run the command 'mocha --timeout 50000', the 50000 ms timeout is to give enough time for tests to complete as they might not process before timeout. 
* On older computers run the command 'mocha --timeout 300000', the 300000 ms timeout is to give enough time for tests to complete as they might not process before timeout on older computers. 
* If need be more than 300000 ms should be used to run the tests depending on the processing power of the computer. 

##Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.
