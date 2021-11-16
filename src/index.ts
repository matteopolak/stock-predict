'use strict';

import * as tf from '@tensorflow/tfjs-node';
import colours from 'colors/safe.js';
import {
	fetchTickerHistory,
	fetchTickerMetadata,
	calculateSimpleMovingAverage,
	splitData,
	trainModel,
	predict
} from './utils.js';

tf.enableProdMode();

const formatter = new Intl.NumberFormat();
const tickerRaw = process.argv[2] ?? 'TSLA';

process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Fetching data for ${colours.bold(colours.white(tickerRaw))}\r`);

const company = await fetchTickerMetadata(tickerRaw);

if (company === null) {
	console.log(`     ${colours.bold(colours.red('X'))}  The ticker ${colours.bold(colours.white(tickerRaw))} is invalid`);

	process.exit(1);
}

const { content: history, size } = await fetchTickerHistory(company.ticker);

console.log(`     ${colours.bold(colours.green('✓'))}  Fetched ${colours.italic(`${history!.length.toString()} entries`)} (${(size / 1024).toFixed(2)} KB) for ${colours.bold(colours.white(company.ticker))} (${colours.bold(colours.white(company.name))})`);

const [ train, _ ] = splitData(
	calculateSimpleMovingAverage(
		history!
	),
	1
);

const inputs: number[][] = [];
const outputs: number[] = [];

for (const [i, data] of train.entries()) {
	inputs.push(data.slice.map(s => s.adjClose));

	if (i + 1 < train.length)
		outputs.push(train[i + 1].slice.at(-1)!.adjClose);
}

const estimate = inputs.pop()!;

const { model } = await trainModel(inputs, outputs);

const name = `${company.ticker.toLowerCase()}_${Date.now()}`;
const path = `${process.cwd().replaceAll('\\\\', '\\')}\\models\\${name}`;
const tomorrow = new Date(Date.now() + 86400000).toISOString().slice(0, 10);

process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Saving model to ${colours.bold(colours.white(path))}\r`);
await model.save(`file://./models/${name}`);
console.log(`     ${colours.bold(colours.green('✓'))}  Saved model to ${colours.bold(colours.white(path))} `);

process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Predicting price of ${colours.bold(colours.white(company.ticker))} for ${colours.bold(colours.white(tomorrow))}\r`);
const prediction = await predict(model, [ estimate ]);
console.log(`     ${colours.bold(colours.green('✓'))}  The price of ${colours.bold(colours.white(company.ticker))} for ${colours.bold(colours.white(tomorrow))} is estimated at ${colours.bold(`${colours.white(`$`)}${colours.green(formatter.format(prediction[0]))}`)}`);