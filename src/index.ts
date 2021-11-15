import { fetchTickerHistory, calculateSimpleMovingAverage, splitData, trainModel, predict } from './utils.js';

const [ train, test ] = splitData(
	calculateSimpleMovingAverage(
		await fetchTickerHistory('TSLA')
	)
);

const inputs = train.map(e => e.slice.map(s => s.close));
const outputs = train.map(e => e.average);

const { model, history } = await trainModel(inputs, outputs);

console.log(await predict(model, test.map(e => e.slice.map(s => s.close))));