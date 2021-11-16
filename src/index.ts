'use strict';

import * as tf from '@tensorflow/tfjs-node-gpu';
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

const tickerRaw = process.argv[2] ?? 'TSLA';

process.stdout.write(`     ${colours.bold(colours.yellow('…'))}  Fetching data for ${colours.bold(colours.white(tickerRaw))}\r`);

const company = await fetchTickerMetadata(tickerRaw);

if (company === null) {
	console.log(`     ${colours.bold(colours.red('X'))}  The ticker ${colours.bold(colours.white(tickerRaw))} is invalid`);

	process.exit(1);
}

const { content: history, size } = await fetchTickerHistory(company.ticker);

console.log(`     ${colours.bold(colours.green('✓'))}  Fetched ${colours.italic(`${history!.length.toString()} entries`)} (${(size / 1024).toFixed(2)} KB) for ${colours.bold(colours.white(company.ticker))} (${colours.bold(colours.white(company.name))})`);

const [ train, test ] = splitData(
	calculateSimpleMovingAverage(
		history!
	),
	1
);

const estimate = train.pop()!;

const inputs = train.map(e => e.slice.map(s => s.adjClose));
const outputs = train.map(e => e.average);

const { model } = await trainModel(inputs, outputs);

await model.save(`file://./models/${company.ticker.toLowerCase()}_${Date.now()}`);

console.log(await predict(model, [ estimate.slice.map(s => s.adjClose) ]));