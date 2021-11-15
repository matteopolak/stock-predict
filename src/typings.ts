export type TickerDay = {
	date: Date,
	open: number,
	high: number,
	low: number,
	close: number,
	adjusted_close: number,
	volume: number
};

export type TickerDayRaw = {
	date: string,
	open: number,
	high: number,
	low: number,
	close: number,
	adjusted_close: number,
	volume: number
};

export type TickerSMACollection = {
	slice: TickerDay[],
	average: number
};