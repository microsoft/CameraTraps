require('dotenv').config();
const path = require('path');
const { API_KEY } = process.env;

module.exports = {
  API_KEY,
  URL: 'https://aiforearth.azure-api.net/api/v1/camera-trap/sync/detect',
  CONFIDENCE: 0.8,
  RENDER_BOXES: false,
  SAMPLE_IMG: path.join('..', '..', 'sample_input', 'test_images',
    'S1_D04_R6_PICT0020.JPG'),
  ALERT: {
    NO_API_KEY: 'No API key found. Please create a .env file in this ' +
      'directory and assign your API key to a variable named "API_KEY" ' +
      'within it.',
  } 
};