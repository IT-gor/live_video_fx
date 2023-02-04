var deltaMin = 128;
var deltaMax = 0;
var deltaAvg = 0;
var deltaArray = new Array(100).fill(0);  // list to build avg of
var deltaCounter = 0;

function getDeltaAvg() {
    let sum = 0;
    for (let i = 0; i < deltaArray.length; i++) {
        sum += deltaArray[i];
    }
    return average = sum / deltaArray.length
}