// One entry of ticker data response object structure from the API
export interface TickerDay {
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

// This is an object in case more data should be provided for each point
export interface TickerCollection {
	slice: TickerDay[]
};

// Ticker metadata response object structure from the API
export interface TickerMetadata {
	description: string,
	startDate: string,
	endDate: string,
	name: string,
	exchangeCode: string,
	ticker: string
};