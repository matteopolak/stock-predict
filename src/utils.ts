'use strict';

import axios from 'axios';
import colours from 'colors';
import * as tf from '@tensorflow/tfjs-node';
import {
	DATA_ENDPOINT_URL,
	METADATA_ENDPOINT_URL,
	WINDOW_SIZE,
	NUM_HIDDEN_LAYERS,
	LEARNING_RATE,
	NUM_EPOCHS,
	INPUT_LAYER_NEURONS,
	INPUT_LAYER_SHAPE,
	RNN_INPUT_SHAPE,
	RNN_OUTPUT_NEURONS,
	OUTPUT_LAYER_NEURONS,
	OUTPUT_LAYER_SHAPE,
	RNN_BATCH_SIZE
} from './constants.js';

import type { Sequential, Tensor, Rank } from '@tensorflow/tfjs-node';
import type { TickerDay, TickerCollection, TickerMetadata } from './typings.js';

// Creates a URL with the ticker from which to fetch stock history data
export function formatDataEndpointURL(ticker: string): string {
  return DATA_ENDPOINT_URL.replace('{ticker}', ticker);
}

// Creates a URL with the ticker from which to fetch ticker metadata
export function formatMetadataEndpointURL(ticker: string): string {
	return METADATA_ENDPOINT_URL.replace('{ticker}', ticker);
}

// Fetches ticker metadata (description, name, start & end dates)
export async function fetchTickerMetadata(ticker: string): Promise<TickerMetadata | null> {
	const response = await axios
		.get<TickerMetadata>(formatMetadataEndpointURL(ticker))
		.catch(() => null);

	return response === null ? null
		: response.data;
}

// Fetch stock data of a ticker
export async function fetchTickerHistory(ticker: string): Promise<{ content: TickerDay[] | null, size: number }> {
	const response = await axios
		.get<ArrayBuffer>(formatDataEndpointURL(ticker), { responseType: 'arraybuffer' })
		.catch(() => null);

	if (response === null)
		return { content: null, size: 0 };

	const json: TickerDay[] = JSON.parse(Buffer.from(response.data).toString());

	for (const day of json) {
		day.date = new Date(day.date);
	}

	return {
		content: json,
		size: response.data.byteLength
	};
}

// Create a window of windowSize length, where each window is shifted by the index
export function createMovingWindow(data: TickerDay[], windowSize: number = WINDOW_SIZE): TickerCollection[] {
	const computed: TickerCollection[] = [];

	for (let i = 0; i <= data.length - windowSize; ++i) {
		computed.push({
			slice: data.slice(i, i + windowSize)
		});
	}

	return computed;
}

// By default, 80% of data will be used for training, and 20% for testing
export function splitData(data: TickerCollection[], splitPercentage: number = 0.8): [ TickerCollection[], TickerCollection[] ] {
	const first = Math.floor(data.length * splitPercentage);

	return [
		data.slice(0, first),
		data.slice(first)
	];
}

// Train the model
export async function trainModel(inputs: number[][], outputs: number[]) {
	const model = tf.sequential();

	// Divide input by 10. Smaller numbers are generally easier to deal with
	const xs = tf.div(tf.tensor2d(inputs, [ inputs.length, inputs[0].length ]), tf.scalar(10));
	const ys = tf.div(tf.reshape(tf.tensor2d(outputs, [ outputs.length, 1 ]), [ outputs.length, 1 ]), tf.scalar(10));

	// Add a `dense` layer (a layer where each node receives input from every node before it)
	model.add(tf.layers.dense({
		units: INPUT_LAYER_NEURONS,
		inputShape: [ INPUT_LAYER_SHAPE ]
	}));

	// Add a `reshape` layer (a layer that shapes the data into a new shape, in this case `RNN_INPUT_SHAPE`)
	model.add(tf.layers.reshape({
		targetShape: RNN_INPUT_SHAPE
	}));

	const lstmCells = [];

	for (let i = 0; i < NUM_HIDDEN_LAYERS; ++i) {
		lstmCells.push(tf.layers.lstmCell({ units: RNN_OUTPUT_NEURONS }));
	}

	// Add a `recurrent neural network` layer (https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)
	model.add(tf.layers.rnn({
		cell: lstmCells,
		inputShape: RNN_INPUT_SHAPE,
		returnSequences: false
	}));

	// Add another dense layer
	model.add(tf.layers.dense({
		units: OUTPUT_LAYER_NEURONS,
		inputShape: [ OUTPUT_LAYER_SHAPE ]
	}));

	// Compile the model with the Adam optimizer, with a loss calculator of meanSquaredError
	// To put it simply, a meanSquaredError loss makes the loss calculation exponentially higher the
	// more wrong it is, as opposed to linear
	model.compile({
		optimizer: tf.train.adam(LEARNING_RATE),
		loss: 'meanSquaredError'
	});

	let currentEpoch = 1;

	// Solely used to make the console look nice
	const batchSize = Math.ceil(inputs.length / RNN_BATCH_SIZE);
	const epochNumberLength = calculateDigitCount(NUM_EPOCHS);
	const batchNumberLength = calculateDigitCount(batchSize);

	// Fit the model to the input data
	const history = await model.fit(xs, ys, {
		batchSize: RNN_BATCH_SIZE,
		epochs: NUM_EPOCHS,
		verbose: 0,
		callbacks: {
			onBatchBegin: batch => {
				process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Epoch ${colours.bold(colours.white(`#${fillZeros(currentEpoch, epochNumberLength)}`))}/${NUM_EPOCHS} | Batch ${colours.bold(colours.white(`#${fillZeros(batch + 1, batchNumberLength)}`))}/${batchSize}\r`);
			},
			onEpochBegin: epoch => {
				currentEpoch = epoch + 1;

				process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Epoch ${colours.bold(colours.white(`#${fillZeros(currentEpoch, epochNumberLength)}`))}/${NUM_EPOCHS} | Batch ${colours.bold(colours.white(`#${fillZeros(1, batchNumberLength)}`))}/${batchSize}\r`);
			},
			onEpochEnd: () => {
				process.stdout.write(`     ${colours.bold(colours.green('✓'))}  Epoch ${colours.bold(colours.white(`#${fillZeros(currentEpoch, epochNumberLength)}`))}/${NUM_EPOCHS} | Batch ${colours.bold(colours.white(`#${batchSize}`))}/${batchSize}\n`);
			}
		}
	});

	return {
		model, history
	};
}

export async function predict(model: Sequential, inputs: number[][]) {
	// Divide input data by 10 (to follow format of model training)
	const tensor = tf.div(
		tf.tensor2d(inputs, [ inputs.length, inputs[0].length ]),
		tf.scalar(10)
	);

	// Predict the next price based off of previous data
	const prediction: Tensor<Rank> = model.predict(tensor) as Tensor<Rank>;

	// Multiply output by 10, to reflect stock price
	return [...tf.mul(prediction, 10).dataSync()];
}

// Log base 10 of a number (floored) will return the # of digits
// since the exponent of 10 when the number system is base 10 is the
// number of digits
export function calculateDigitCount(number: number) {
	return Math.floor(Math.log10(number));
}

// Solely visual, just fills zeros in front of a number
export function fillZeros(number: number, length: number) {
	return `${'0'.repeat(length - calculateDigitCount(number))}${number}`;
}