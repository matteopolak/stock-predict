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
import type { TickerDay, TickerSMACollection, TickerMetadata } from './typings.js';

export function formatDataEndpointURL(ticker: string): string {
  return DATA_ENDPOINT_URL.replace('{ticker}', ticker);
}

export function formatMetadataEndpointURL(ticker: string): string {
	return METADATA_ENDPOINT_URL.replace('{ticker}', ticker);
}

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

// Calculate the SMA (simple moving average) of each data point
export function calculateSimpleMovingAverage(data: TickerDay[], windowSize: number = WINDOW_SIZE): TickerSMACollection[] {
	const computed: TickerSMACollection[] = [];

	for (let i = 0; i <= data.length - windowSize; ++i) {
		let current = 0;

		const boundary = i + windowSize;

		for (let x = i; x < boundary && x <= data.length; ++x) {
			current += data[i].adjClose / windowSize;
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
	const model = tf.sequential();

	const xs = tf.div(tf.tensor2d(inputs, [ inputs.length, inputs[0].length ]), tf.scalar(10));
	const ys = tf.div(tf.reshape(tf.tensor2d(outputs, [ outputs.length, 1 ]), [ outputs.length, 1 ]), tf.scalar(10));

	model.add(tf.layers.dense({
		units: INPUT_LAYER_NEURONS,
		inputShape: [ INPUT_LAYER_SHAPE ]
	}));

	model.add(tf.layers.reshape({
		targetShape: RNN_INPUT_SHAPE
	}));

	const lstmCells = [];

	for (let i = 0; i < NUM_HIDDEN_LAYERS; ++i) {
		lstmCells.push(tf.layers.lstmCell({ units: RNN_OUTPUT_NEURONS }));
	}

	model.add(tf.layers.rnn({
		cell: lstmCells,
		inputShape: RNN_INPUT_SHAPE,
		returnSequences: false
	}));

	model.add(tf.layers.dense({
		units: OUTPUT_LAYER_NEURONS,
		inputShape: [ OUTPUT_LAYER_SHAPE ]
	}));

	model.compile({
		optimizer: tf.train.adam(LEARNING_RATE),
		loss: 'meanSquaredError'
	});

	let currentEpoch = 1;

	const batchSize = Math.ceil(inputs.length / RNN_BATCH_SIZE);

	const epochNumberLength = calculateDigitCount(NUM_EPOCHS);
	const batchNumberLength = calculateDigitCount(batchSize);

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
	const tensor = tf.div(
		tf.tensor2d(inputs, [ inputs.length, inputs[0].length ]),
		tf.scalar(10)
	);

	console.log('predict');
	const prediction: Tensor<Rank> = model.predict(tensor) as Tensor<Rank>;

	return [...tf.mul(prediction, 10).dataSync()];
}

export function calculateDigitCount(number: number) {
	return Math.floor(Math.log10(number));
}

export function fillZeros(number: number, length: number) {
	return `${'0'.repeat(length - calculateDigitCount(number))}${number}`;
}