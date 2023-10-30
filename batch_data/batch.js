// File to divide the original data to batches of 100 words
const fs = require('fs');
const filePath = 'WLASL_v0.3.json';

fs.readFile(filePath, 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading the file:', err);
    return;
  }
  try {
    const jsonData = JSON.parse(data);
    const chunkSize = 100;
    for (let i = 0; i < jsonData.length; i += chunkSize) {
        const chunk = JSON.stringify(jsonData.slice(i, i + chunkSize));
        const fileName = `batch_${i / chunkSize + 1}.json`;
        try{
          fs.writeFileSync(fileName, chunk);
          console.log(`File ${fileName} created with ${chunk.length} elements.`);
        }
        catch(err){
          console.error(err);
        }
    }
  } catch (error) {
    console.error('Error parsing JSON:', error);
  }
});
