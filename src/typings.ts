export type TickerDay = {
	date: Date,
	close: number,
	high: number,
	low: number,
	open: number,
	volume: number,
	adjClose: number,
	adjHigh: number,
	adjLow: number,
	adjOpen: number,
	adjVolume: number,
	divCash: number,
	splitFactor: number
};

export type TickerSMACollection = {
	slice: TickerDay[],
	average: number
};

export type TickerMetadata = {
	description: string,
	startDate: string,
	endDate: string,
	name: string,
	exchangeCode: string,
	ticker: string
};