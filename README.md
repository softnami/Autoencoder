# Autoencoder 
###[Author: Hussain Mir Ali]
An autoencoder neural network with single hidden layer and multiclass ouput. This project has been written in JavaScript. 

##External Librarbies Used:
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
*  Copy the 'Autoencoder' folder to your project directory.


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

```
```
<!--index.html-->
<!doctype html>
<html>
  <head>
  </head>
  <body >
        <script src="Autoencoder/lib/q.js"></script>
        <script src="Autoencoder/lib/math.js"></script>
        <script src="Autoencoder/Autoencoder.js"></script>
</body>
</html>

*/
```
