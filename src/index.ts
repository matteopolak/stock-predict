'use strict';

import * as tf from '@tensorflow/tfjs-node';
import colours from 'colors/safe.js';
import {
	fetchTickerHistory,
	fetchTickerMetadata,
	splitData,
	trainModel,
	predict,
	createMovingWindow,
	formatBytes,
	formatter
} from './utils.js';

tf.enableProdMode();

console.time('train');

// Get the ticker from CLI args, use TSLA as default
const tickerRaw = process.argv[2] ?? 'TSLA';

process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Fetching data for ${colours.bold(colours.white(tickerRaw))}\r`);

// Fetch ticker information (not needed, just looks nice)
const company = await fetchTickerMetadata(tickerRaw);

if (company === null) {
	console.log(`     ${colours.bold(colours.red('X'))}  The ticker ${colours.bold(colours.white(tickerRaw))} is invalid`);

	process.exit(1);
}

// Fetch all stock price history for the ticker
const { content: history, size } = await fetchTickerHistory(company.ticker);

console.log(`     ${colours.bold(colours.green('✓'))}  Fetched ${colours.italic(`${history!.length.toString()} entries`)} (${formatBytes(size)}) for ${colours.bold(colours.white(`${company.exchangeCode}:${company.ticker}`))} (${colours.bold(colours.white(company.name))})`);

// Create moving window (like SMA without the average) for each data point
const [ train, _ ] = splitData(
	createMovingWindow(
		history!
	),
	1
);

const inputs: number[][] = [];
const outputs: number[] = [];

// Format input for the model
for (const [i, data] of train.entries()) {
	inputs.push(data.slice.map(s => s.adjClose));

	if (i + 1 < train.length)
		outputs.push(train[i + 1].slice.at(-1)!.adjClose);
}

// Remove the most recent one, as it's the one we'll use to predict the next price
const estimate = inputs.pop()!;

// Train the model
const { model, time } = await trainModel(inputs, outputs);

const name = `${company.ticker.toLowerCase()}_${Date.now()}`;
const path = `${process.cwd().replaceAll('\\\\', '\\')}\\models\\${name}`;
const tomorrow = new Date(history!.at(-1)!?.date.getTime() + 86400000).toISOString().slice(0, 10);

console.log(`     ${colours.bold(colours.green('✓'))}  Trained model in ${colours.bold(colours.white(`${formatter.format(time / 60000)} minutes`))}`);
process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Saving model to ${colours.bold(colours.white(path))}\r`);

// Save the model
await model.save(`file://./models/${name}`);

console.log(`     ${colours.bold(colours.green('✓'))}  Saved model to ${colours.bold(colours.white(path))} `);
process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Predicting price of ${colours.bold(colours.white(`${company.exchangeCode}:${company.ticker}`))} for ${colours.bold(colours.white(tomorrow))}\r`);

// Predict the next price
const prediction = await predict(model, [ estimate ]);
console.log(`     ${colours.bold(colours.green('✓'))}  The price of ${colours.bold(colours.white(`${company.exchangeCode}:${company.ticker}`))} for ${colours.bold(colours.white(`${tomorrow} EOD`))} is estimated at ${colours.bold(`${colours.white(`$`)}${colours.green(formatter.format(prediction[0]))}`)}`);