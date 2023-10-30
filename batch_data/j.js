// File to divide the original data to batches of 100 words
const fs = require('fs');
const filePath = 'batch_data/batch_1.json';

fs.readFile(filePath, 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading the file:', err);
    return;
  }
  try {
    const jsonData = JSON.parse(data);
    console.log(jsonData?jsonData.length:"no data");
  } catch (error) {
    console.error('Error parsing JSON:', error);
  }
});
