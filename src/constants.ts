'use strict';

export const DATA_ENDPOINT_URL: string = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=1000-1-1&endDate=9999-1-1&token=7f5d93a97c46f7b3f8ad0e8b0a770031822e81ee';
export const METADATA_ENDPOINT_URL: string = 'https://api.tiingo.com/tiingo/daily/{ticker}?token=7f5d93a97c46f7b3f8ad0e8b0a770031822e81ee';
export const WINDOW_SIZE: number = 12;

export const NUM_EPOCHS: number = 100;
export const LEARNING_RATE: number = 0.0016;
export const NUM_HIDDEN_LAYERS: number = 3;
export const NUM_ITEMS_PERCENT: number = 50;
export const NUM_LAYER_NEURONS: number = 100;

export const INPUT_LAYER_SHAPE: number = WINDOW_SIZE;
export const INPUT_LAYER_NEURONS: number = NUM_LAYER_NEURONS;
export const RNN_INPUT_LAYER_FEATURES: number = 10;
export const RNN_INPUT_LAYER_TIMESTEPS: number = INPUT_LAYER_NEURONS / RNN_INPUT_LAYER_FEATURES;
export const RNN_INPUT_SHAPE: [ number, number ] = [ RNN_INPUT_LAYER_FEATURES, RNN_INPUT_LAYER_TIMESTEPS ];
export const RNN_OUTPUT_NEURONS: number = 20;
export const RNN_BATCH_SIZE: number = WINDOW_SIZE;
export const OUTPUT_LAYER_SHAPE: number = RNN_OUTPUT_NEURONS;
export const OUTPUT_LAYER_NEURONS: number = 1;