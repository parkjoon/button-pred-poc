import './App.css';
import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const trainingData = [
  // {
  //   x: [ [pageX, pageY, deltaX, deltaY, deltaT, timeStamp] ], // Slice out timeStamp when inputting to RNN
  //   y: 0
  // }
  {
    x: [],
    y: null
  }
];

const handleOnMouseMove = (e) => {
  const arr = trainingData[trainingData.length - 1].x;
  const prev = arr.length ? arr[arr.length - 1] : [e.pageX, e.pageY, e.timeStamp];
  arr.push([
    e.pageX,
    e.pageY,
    prev[0] - e.pageX,
    prev[1] - e.pageY,
    e.timeStamp - prev[2],
    e.timeStamp
  ]);
}
document.addEventListener('mousemove', handleOnMouseMove, false);

let input = tf.layers.simpleRNN({
  inputShape: [100, 5],
  units: 50
});
let output = tf.layers.dense({
  units: 4,
  activation: 'softmax'
});
let model = tf.sequential();
model.add(input);
model.add(output);
model.compile({
  loss: 'sparseCategoricalCrossentropy',
  optimizer: 'adamax',
  metrics: ['accuracy']
});

function App() {
  const [ prediction, setPrediction ] = useState();
  const [ listenerState, setListenerState ] = useState("Collecting mouse data...");

  const updateModel = () => {
    console.log("Training Data before updating model:", trainingData);

    const batchX = [];
    const batchY = [];

    while (trainingData.length > 1) {
      const dataToBatch = trainingData.shift();
      batchX.push(dataToBatch.x.map(x => x.slice(0,5)));
      batchY.push(dataToBatch.y);
    }

    const x = tf.tensor3d(batchX);
    const y = tf.tensor1d(batchY, "float32");

    console.log("batchX", x, Array.from(batchX));
    console.log("batchY", y, Array.from(batchY));

    model.fit(x, y, {
      epochs: 200,
      validationSplit: 0.2,
      callbacks: {
        onTrainStart: setListenerState("Updating model..."),
        onTrainEnd: setListenerState("Finished updating model!")
      }
    })
  }

  const predict = async () => {
    console.log("Predicting...");
    const recentSequence = tf.tensor3d([trainingData[trainingData.length - 1].x.slice(0,100).map(x => x.slice(0,5))]);
    const probabilityArray = await model.predict(recentSequence).data();
    console.log("Prediction:", probabilityArray);
    setPrediction(probabilityArray);
  }

  const handleButtonClick = (id) => {
    const lastEntry = trainingData[trainingData.length - 1];

    // TODO
    if (lastEntry.x.length > 100) {
      // const skipCount = Math.ceil(lastEntry.x.length / 100);
      // lastEntry.x = lastEntry.x.filter((e, i) => i % skipCount === skipCount - 1);
      lastEntry.x = trainingData[trainingData.length - 1].x.slice(0,100);
    }

    lastEntry.y = id.charCodeAt(0) - 97;
    trainingData.push({
      x: [],
      y: null
    });
  }

  const handleKeyDown = (e) => {
    if (e.keyCode === 85) { // u
      updateModel();
    } else if (e.keyCode === 80) { // p
      predict();
    }
  }

  return (
    <div className="App" onKeyDown={handleKeyDown}>
      <div style={{ position: 'absolute', top: 100, left: 0, width: '100%', textAlign: 'center' }}>
        <div>Press the 'u' key to update the model.</div>
        <br />
        <div>
          <div>Press the 'p' key to evaluate a prediction:</div>
          <ol>
            {prediction?.length ? Array.from(prediction).map((p, i) => {
              return (
                <li key={i}>
                  BUTTON {String.fromCharCode(97 + i).toUpperCase()}: {(p*100).toFixed(1)}%
                </li>
              );
            }) : <li>{listenerState}</li>}
          </ol>
        </div>
      </div>

      <button onClick={handleButtonClick.bind(this, 'a')} className="buttonA">BUTTON A</button>
      <button onClick={handleButtonClick.bind(this, 'b')} className="buttonB">BUTTON B</button>
      <button onClick={handleButtonClick.bind(this, 'c')} className="buttonC">BUTTON C</button>
      <button onClick={handleButtonClick.bind(this, 'd')} className="buttonD">BUTTON D</button>
    </div>
  );
}

export default App;
