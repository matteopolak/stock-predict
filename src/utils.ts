import axios from 'axios';
import * as tf from '@tensorflow/tfjs-node';

import { DATA_ENDPOINT_URL, WINDOW_SIZE, NUM_LAYER_NEURONS, NUM_HIDDEN_LAYERS, LEARNING_RATE, NUM_EPOCHS } from './constants.js';

import type { Sequential, Tensor, Rank } from '@tensorflow/tfjs-node';
import type { TickerDay, TickerDayRaw, TickerSMACollection } from './typings.js';

export function formatDataEndpointURL(ticker: string): string {
  return DATA_ENDPOINT_URL.replace('{ticker}', ticker);
}

// Fetch stock data of a ticker
export async function fetchTickerHistory(ticker: string): Promise<TickerDay[]> {
	const { data } = await axios.get<TickerDayRaw[]>(formatDataEndpointURL(ticker));

	return data.map<TickerDay>(day => ({
		...day,
		date: new Date(day.date)
	}));
}

// Calculate the SMA (simple moving average) of each data point
export function calculateSimpleMovingAverage(data: TickerDay[], windowSize: number = WINDOW_SIZE): TickerSMACollection[] {
	const computed: TickerSMACollection[] = [];

	for (const [i, entry] of data.entries()) {
		let current = 0;

		const boundary = i + windowSize;

		for (let x = i; x < boundary && x <= data.length; ++x) {
			current += entry.close / windowSize;
		}

		computed.push({
			slice: data.slice(i, i + windowSize),
			average: current
		});
	}

	return computed;
}

// By default, 80% of data will be used for training, and 20% for testing
export function splitData(data: TickerSMACollection[], splitPercentage: number = 0.8): [ TickerSMACollection[], TickerSMACollection[] ] {
	const first = Math.floor(data.length * splitPercentage);

	return [
		data.slice(0, first),
		data.slice(first)
	];
}

export async function trainModel(inputs: number[][], outputs: number[]) {
	const inputLayerShape = WINDOW_SIZE;
	const inputLayerNeurons = NUM_LAYER_NEURONS;
	
	const rnnInputLayerFeatures = 10;
	const rnnInputLayerTimesteps = inputLayerNeurons / rnnInputLayerFeatures;

	const rnnInputShape = [ rnnInputLayerFeatures, rnnInputLayerTimesteps ];
	const rnnOutputNeurons = 20;
	const rnnBatchSize = WINDOW_SIZE;
	
	const outputLayerShape = rnnOutputNeurons;
	const outputLayerNeurons = 1;

	const model = tf.sequential();

	const xs = tf.tensor2d(inputs, [ inputs.length, inputs[0].length ]).div(tf.scalar(10));
	const ys = tf.tensor2d(outputs, [ outputs.length, 1 ]).reshape([ outputs.length, 1 ]).div(tf.scalar(10));

	model.add(tf.layers.dense({
		units: inputLayerNeurons,
		inputShape: [ inputLayerShape ]
	}));

	model.add(tf.layers.reshape({
		targetShape: rnnInputShape
	}));

	const lstmCells = [];

	for (let i = 0; i < NUM_HIDDEN_LAYERS; ++i) {
		lstmCells.push(tf.layers.lstmCell({ units: rnnOutputNeurons }));
	}

	model.add(tf.layers.rnn({
		cell: lstmCells,
		inputShape: rnnInputShape,
		returnSequences: false
	}));

	model.add(tf.layers.dense({
		units: outputLayerNeurons,
		inputShape: [ outputLayerShape ]
	}));

	model.compile({
		optimizer: tf.train.adam(LEARNING_RATE),
		loss: 'meanSquaredError'
	});

	const history = await model.fit(xs, ys, {
		batchSize: rnnBatchSize,
		epochs: NUM_EPOCHS
	});

	return {
		model, history
	};
}

export async function predict(model: Sequential, inputs: number[][]) {
	const tensor = tf
		.tensor2d(inputs, [ inputs.length, inputs[0].length ])
		.div(tf.scalar(10));

	const prediction: Tensor<Rank> = model.predict(tensor) as Tensor<Rank>;

	return [...tf.mul(prediction, 10).dataSync()];
}