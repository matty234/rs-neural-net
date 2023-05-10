use ndarray::{Array, Array2, ArrayView2, Axis};
use rand::Rng;

use std::{
    borrow::Borrow,
    fmt::{Debug, Formatter},
    fs::{self, File},
    ops::{Add, Mul},
    path::PathBuf,
};
mod mnistparser;

use ndarray::prelude::*;

use crate::mnistparser::MnistImage;

type ActivationFunction = fn(ArrayView2<f64>) -> Array2<f64>;

enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
}

impl Activation {
    fn sigmoid(x: ArrayView2<f64>) -> Array2<f64> {
        x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sigmoid_derivative(x: ArrayView2<f64>) -> Array2<f64> {
        Self::sigmoid(x).mapv(|x| x * (1.0 - x))
    }

    fn tanh(x: ArrayView2<f64>) -> Array2<f64> {
        x.mapv(|x| x.tanh())
    }

    fn tanh_derivative(x: ArrayView2<f64>) -> Array2<f64> {
        x.mapv(|x| 1.0 - x.tanh().powi(2))
    }

    fn relu(x: ArrayView2<f64>) -> Array2<f64> {
        x.mapv(|x| x.max(0.0))
    }

    fn relu_derivative(x: ArrayView2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn get_function(&self) -> ActivationFunction {
        match self {
            Activation::Sigmoid => Self::sigmoid,
            Activation::Tanh => Self::tanh,
            Activation::ReLU => Self::relu,
        }
    }

    fn get_derivative(&self) -> ActivationFunction {
        match self {
            Activation::Sigmoid => Self::sigmoid_derivative,
            Activation::Tanh => Self::tanh_derivative,
            Activation::ReLU => Self::relu_derivative,
        }
    }
}

struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    fn new(layers: Vec<Box<dyn Layer>>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }

    // performs forward pass
    fn feed_forward(&self, input: ArrayView2<f64>) -> Vec<ForwardResult> {
        let mut v = Vec::new();

        println!("input shape: {:?}", input.shape());
        // this is in effect A_0 (or X)
        v.push(ForwardResult {
            activation: input.to_owned(),
            pre_activation: input.to_owned(),
        });
        for (i, layer) in self.layers.iter().enumerate() {
            println!("layer: {:?}", i);
            println!("output so far: {:?}", v[i].activation);

            let A = v[i].activation.view();
            // use previous output as input

            let next_a = layer.forward(A);
            println!("next_a shape: {:?}", next_a);
            v.push(next_a);

            // set output to the current output
            //println!("output shape: {:?} \n", output.shape());
        }

        v
    }

    // returns layer deltas (A, dZ)
    fn backprop(
        &mut self,
        input: ArrayView2<f64>,
        error: ArrayView2<f64>,
        reveresed_layer_outputs: Vec<ForwardResult>,
    ) -> Vec<BackwardsError> {
        let mut layer_errors = Vec::new();
        let mut layer_error = error.to_owned();
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            // ? = layer.backward(A, dZ)
            let le = layer.backward(
                input,
                layer_error.view(),
                reveresed_layer_outputs[i].pre_activation.view(),
                reveresed_layer_outputs[i + 1].activation.view(),
            );
            layer_error = le.actual.clone();
            layer_errors.push(le);
        }
        layer_errors
    }

    // updates weights and biases
    fn update(
        &mut self,
        layer_errors: Vec<BackwardsError>,
        input: ArrayView2<f64>,
        learning_rate: f64,
    ) {
        for (layer_pos, layer) in self.layers.iter_mut().enumerate() {
            layer.update(layer_errors[layer_pos].clone(), input, learning_rate);
        }
    }

    fn get_ratio_correct(&self, input: ArrayView2<f64>, labels: Vec<usize>) -> f64 {
        let mut correct = 0;
        let mut total = 0;
        for (i, x) in input.outer_iter().enumerate() {
            println!("x shape: {:?}", x.shape());
            // create 2d array from 1d array
            let x = x.into_shape((1, x.len())).unwrap();
            let prediction = self.predict(x.view());

            let actual_pos = labels[i];
            if prediction == actual_pos {
                correct += 1;
            }
            total += 1;
        }
        correct as f64 / total as f64
    }

    fn train(
        &mut self,
        input: ArrayView2<f64>,
        target: ArrayView2<f64>,
        learning_rate: f64,
        labels: Vec<usize>,
    ) {
        let mut output = self.feed_forward(input);

        //println!("target shape: {:?}", target);

        let current_error = output[output.len() - 1].activation.clone() - target;
        //println!("current error: {:?}", current_error);

        output.reverse();

        let mut layer_errors = self.backprop(input, current_error.view(), output);

        layer_errors.reverse();
        self.update(layer_errors, input, learning_rate);
        println!();

        //println!("ratio correct: {}", self.get_ratio_correct(input, labels))
    }

    fn predict(&self, input: ArrayView2<f64>) -> usize {
        let results = self.feed_forward(input);
        let resultvec = results[results.len() - 1].activation.clone();
        //println!("resultvec: {:?}", resultvec);
        let mut max = 0.0;
        let mut max_index = 0;
        for (i, x) in resultvec.iter().enumerate() {
            if *x > max {
                max = *x;
                max_index = i;
            }
        }
        max_index
    }
}

#[derive(Debug, Clone)]
struct BackwardsError {
    delta_bias: Array1<f64>,
    delta_weight: Array2<f64>,
    actual: Array2<f64>,
}

#[derive(Clone)]
struct ForwardResult {
    // A
    activation: Array2<f64>,

    // Z
    pre_activation: Array2<f64>,
}

impl Debug for ForwardResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForwardResult")
            .field("A Shape", &self.activation.shape())
            .field("Z Shape", &self.pre_activation.shape())
            .field("A", &self.activation)
            .field("Z", &self.pre_activation)
            .finish()
    }
}

trait Layer {
    fn forward(&self, input: ArrayView2<f64>) -> ForwardResult;
    fn backward(
        &self,
        input: ArrayView2<f64>,
        error: ArrayView2<f64>,
        preactivated_output: ArrayView2<f64>,
        previous_activation: ArrayView2<f64>,
    ) -> BackwardsError;
    fn update(&mut self, layer_deltas: BackwardsError, input: ArrayView2<f64>, learning_rate: f64);
}

struct DenseLayer {
    weights: Array2<f64>,
    biases: Array2<f64>,
    activation: Activation,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize, activation: Activation) -> DenseLayer {
        let mut rng = rand::thread_rng();
        // setup weights and bias vectors with random values
        let weights = Array::from_shape_fn((output_size, input_size), |(_i, _j)| {
            rng.gen_range(-1.0..1.0)
        });
        let biases = Array::from_shape_fn((output_size, 1), |(_i, _j)| rng.gen_range(-1.0..1.0));
        DenseLayer {
            weights,
            biases,
            activation,
        }
    }
}

impl Layer for DenseLayer {
    // input is A (or X)
    fn forward(&self, input: ArrayView2<f64>) -> ForwardResult {
        let activation_function = self.activation.get_function();

        // Z
        let weighted_sum = self.weights.dot(&input).add(&self.biases);
        //println!("input (after dot) shape: {:?}", weighted_sum.shape());

        // A
        let input_activation = activation_function(weighted_sum.view());

        ForwardResult {
            activation: input_activation,
            pre_activation: weighted_sum,
        }
    }

    // A, dZ (previous error), Z
    fn backward(
        &self,
        input: ArrayView2<f64>,
        error: ArrayView2<f64>,
        preactivated_output: ArrayView2<f64>,
        previous_activation: ArrayView2<f64>,
    ) -> BackwardsError {
        println!("backprop - dense layer");
        let activation_function = self.activation.get_derivative();

        let dZ = error.to_owned() * activation_function(preactivated_output);
        let delta_weights = 1.0 / input.shape()[1] as f64 * dZ.dot(&previous_activation.t());
        let delta_biases = 1.0 / input.shape()[1] as f64 * dZ.sum_axis(Axis(1));

        // we use dZ with the weight in the next layer
        // so we need to transpose the weights
        let error = self.weights.t().dot(&dZ);

        BackwardsError {
            delta_bias: delta_biases,
            delta_weight: delta_weights,
            actual: error,
        }
    }

    fn update(&mut self, layer_deltas: BackwardsError, input: ArrayView2<f64>, learning_rate: f64) {
        println!("update - dense layer");

        println!(
            "layer_deltas.delta_weight shape: {:?}",
            layer_deltas.delta_weight.shape()
        );
        println!(
            "layer_deltas.delta_bias shape: {:?}",
            layer_deltas.delta_bias.shape()
        );

        println!("self.weights shape: {:?}", self.weights.shape());
        println!("self.biases shape: {:?}", self.biases.shape());

        // disabled transpose below

        self.weights = self
            .weights
            .borrow()
            .add(layer_deltas.delta_weight * learning_rate * -1.0);
        self.biases = self
            .biases
            .borrow()
            .t()
            .add(layer_deltas.delta_bias.mul(learning_rate * -1.0));
        self.biases = self.biases.t().to_owned();
        println!("self.biases after update shape: {:?}", self.biases.shape());
        println!(
            "self.weights after update shape: {:?}",
            self.weights.shape()
        );
    }
}

struct OutputLayer {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl OutputLayer {
    fn new(input_size: usize, output_size: usize) -> OutputLayer {
        let mut rng = rand::thread_rng();
        // setup weights and bias vectors with random values
        let weights = Array::from_shape_fn((output_size, input_size), |(_i, _j)| {
            rng.gen_range(-1.0..1.0)
        });
        let biases = Array::from_shape_fn((output_size, 1), |(_i, _j)| rng.gen_range(-1.0..1.0));
        OutputLayer { weights, biases }
    }

    fn softmax(input: ArrayView2<f64>) -> Array2<f64> {
        //println!("input to softmax shape: {:?}", input.shape());
        //println!("input to softmax: {:?}", input);
        // map into a f64 array
        let input = input.mapv(|x| x.exp());

        let input_sum = input.sum_axis(Axis(0));

        let output = input / input_sum;

        return output;
    }
}

impl Layer for OutputLayer {
    fn forward(&self, input: ArrayView2<f64>) -> ForwardResult {
        //println!("output layer weights: {:?}", self.weights);

        // Z
        let weighted_sum = self.weights.dot(&input).add(&self.biases);
        // A

        //println!("output layer weighted sum: {:?}", weighted_sum);
        let input_activation = OutputLayer::softmax(weighted_sum.view());

        ForwardResult {
            activation: input_activation,
            pre_activation: weighted_sum.clone(),
        }
    }

    // A, dZ
    fn backward(
        &self,
        input: ArrayView2<f64>,
        error: ArrayView2<f64>,
        preactivated_output: ArrayView2<f64>,
        previous_activation: ArrayView2<f64>,
    ) -> BackwardsError {
        println!("backprop - output layer");
        let m = input.shape()[1] as f64;

        let dW = (1.0 / input.shape()[1] as f64) * error.dot(&previous_activation.t());
        let dB = (1.0 / input.shape()[1] as f64) * error.sum_axis(Axis(1));

        let err = self.weights.t().dot(&error);
        return BackwardsError {
            delta_weight: dW,
            delta_bias: dB,
            actual: err,
        };
    }

    fn update(&mut self, layer_deltas: BackwardsError, input: ArrayView2<f64>, learning_rate: f64) {
        //println!("update - output layer");
        //println!(
        //    "layer_deltas.delta_weight shape: {:?}",
        //    layer_deltas.delta_weight.shape()
        //);
        //println!(
        //    "layer_deltas.delta_bias shape: {:?}",
        //    layer_deltas.delta_bias.shape()
        //);

        //println!("self.weights shape: {:?}", self.weights.shape());
        //println!("self.biases shape: {:?}", self.biases.shape());

        self.weights = self
            .weights
            .borrow()
            .add(layer_deltas.delta_weight * learning_rate * -1.0);
        self.biases = self
            .biases
            .borrow()
            .t()
            .add(layer_deltas.delta_bias.mul(learning_rate * -1.0));
        self.biases = self.biases.t().to_owned();
    }
}

fn write_image_to_label_folder(first: bool, image: &MnistImage, label: usize, index: usize) {
    let mut path = PathBuf::from("output");

    if first {
        if path.exists() {
            match fs::remove_dir_all(&path) {
                Err(why) => println!("! {:?}", why.kind()),
                Ok(_) => println!("removed folder"),
            }
        }
    }

    path.push(label.to_string());

    // create label subfolder if it doesn't exist
    if !path.exists() {
        match fs::create_dir_all(&path) {
            Err(why) => println!("! {:?}", why.kind()),
            Ok(_) => println!("created folder"),
        }
    }

    path.push(index.to_string() + "-" + &image.label.to_string() + ".png");

    image.write_to_bitmap(path.to_str().unwrap());
}

fn main() {
    let file = File::open("data/train-images-idx3-ubyte.gz").unwrap();
    let file_labels = File::open("data/train-labels-idx1-ubyte.gz").unwrap();

    let mut m = mnistparser::MnistParser::new();
    m.parse(file, file_labels);

    let train_size = 10000;

    let m_train = m.from_mnist_parser_get_range(0, train_size);
    let m_test = m.from_mnist_parser_get_range(train_size, train_size + 100);

    let image_matrix = m_train.get_image_matrix();
    let label_matrix = m_train.get_label_matrix();
    let label_vector = m_train.get_label_vector();

    let mut network = NeuralNetwork::new(vec![
        Box::new(DenseLayer::new(28 * 28, 16, Activation::ReLU)),
        Box::new(OutputLayer::new(16, 10)),
    ]);

    let total_training_iter = 1;
    for i in 0..total_training_iter {
        network.train(
            image_matrix.view(),
            label_matrix.view(),
            0.2,
            label_vector.clone(),
        );

        println!("Iteration: {}, Error: {}", i + 1, 0.0);

        println!()
    }

    println!("Training complete");

    /*let mut correct = 0;
    let mut total = 0;
    let mut iter = 0;
    for (i_image, image) in m_test.images.iter().enumerate() {
        iter += 1;

        let image_vec = image.get_f64_pixels();
        let output = network.predict(image_vec.view());

        if output == image.get_label() as usize {
            correct += 1;
        }
        total += 1;
        println!("Predicted: {}, Actual: {} for image {}", output, image.get_label(), i_image);

        if iter % 10 == 0 {
            println!("Accuracy: {}", correct as f64 / total as f64);
        }

        write_image_to_label_folder(iter==1, image, output, i_image);
    }

    println!("Accuracy: {}", correct as f64 / total as f64);*/
}
