# Examples calling the synchronous API

Test images in `sample_input` are from the Snapshot Serengeti dataset by the Snapshot Serengeti program at the University of Minnesota Lion Center. More of this dataset and others are available at [lila.science](http://lila.science/).

## Calling the API with Python

Refer to the `example.ipynb` notebook for an example of invoking the API with Python.

## Calling the API with Node / JavaScript

To run a sample invocation from Node, navigate to the `node_modules` directory, and perform the following:

### Install dependencies

Run the following command to install node dependencies: 
```
npm install
```

### Add a .env file and save your API key to it

Create a file named `.env` in the `node_modules` directory, and save the your API key to it:
```
# .env contents 

API_KEY=<copy-your-api-key-here>
```

### Run the script to call the API

```
npm start
```
NOTE: to send different images to the API for inference, edit the `SAMPLE_IMG` property in `config.js`.