const fs = require('fs');
const path = require('path');
const util = require('util');
const agent = require('superagent');
const config = require('./config');

const callAPI = async (image) => {
  let res;
  try {
    // For now only requesting on one image at a time 
    // Attaching more images to the request is possible,
    // but you'd have to use callbacks rather than async/await:
    // https://stackoverflow.com/questions/34403670/superagent-multiple-files-attachment
    res = await agent
      .post(config.URL)
      .query({ confidence: config.CONFIDENCE })
      .query({ render: config.RENDER_BOXES })
      .set('Ocp-Apim-Subscription-Key', config.API_KEY)
      .attach(image.name, image.path);
  } catch (err) {
    throw new Error(err);
  }
  return res;
};

const start = async () => {
  // Make sure there's an API key in the .env file
  if (!config.API_KEY) {
    console.log(config.ALERT.NO_API_KEY);
    process.exit(9);
  }
  const sampleImage = {
    name: path.basename(config.SAMPLE_IMG),
    path: config.SAMPLE_IMG,
  };
  try {
    const res = await callAPI(sampleImage);
    const detectionsTmpFile = res.files.detection_result.path;
    const classificationsTmpFile = res.files.classification_result.path;
    const results = {
      detections: JSON.parse(fs.readFileSync(detectionsTmpFile)),
      classifications: JSON.parse(fs.readFileSync(classificationsTmpFile)),
    };
    console.log(util.inspect(results, {showHidden: false, depth: null}));
  } catch(err) {
    console.log(err)
  }
};

start();