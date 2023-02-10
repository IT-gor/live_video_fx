
let attack = 0.01;
let release = 0.1
let buffers = [];
let bufferSourceNodes = [];
var output;
var oldNote = -1;
var oldFreq = 59;
let clearCanvas2 = document.getElementById("clearCanvas");
WIDTH = 640;
HEIGHT = 320;

//Gibt den Output zurück
function getOutput(){
    return this.output;
}

if (navigator.requestMIDIAccess) {
    navigator.requestMIDIAccess({sysex: true}).then(function (midiAccess) {
        midi = midiAccess;
        var inputs = midi.inputs.values();

        // Output festlegen!
        // output = midiAccess.outputs.get('output-1');
        output = midiAccess.outputs.get('6FF5590044F4859ED50C5167BCFE9700A1798E39AA55A628E86D39011FAECD5D');

        // Inputs und Outputs ausgeben:
        listInputsAndOutputs( midiAccess );

        // loop through all inputs
        for (var input = inputs.next(); input && !input.done; input = inputs.next()) {
            // listen for midi messages
            input.value.onmidimessage = onMIDIMessage;
        }


    });
} else {
    alert("No MIDI support in your browser.");
}


function onMIDIMessage(event) {
    
    // event.data is an array
    // event.data[0] = on (144) / off (128) / controlChange (176)  / pitchBend (224) / songpos (242) ...
    // event.data[1] = midi note
    // event.data[2] = velocity
    // console.log("event.data[0]: " + event.data[0])
    // console.log(event)
    // console.log("event.value: " + ((event.data[2] & 0b111111) << 7) + event.data[1])
    // console.log("filter.frequency.value: " + filter.frequency.value)

    switch (event.data[0]) {
    case 144:
        // IG: ausgelagert in ui.js
        // das canvas ist nicht mehr leer!
        canvas_empty = false;
        // console.log("midi case 144, canvas_empty:")
        // console.log(canvas_empty)
        clearCanvas.classList.add('activeButton');
        break;
    case 224:
        // your function pitchBend(LSB, HSB)
        // pitchBend(event.data[1], event.data[2]);
        var y = ((event.data[2] & 0b111111) << 7) + event.data[1];
        // console.log("case 224 pitch / y: " + y)
        if (oldFreq != -1) {
            stopNoteFreq(oldFreq, 127);
        }
        
        // IG: Pixel in x von Frequenz umrechnen = height 480 / anzahl von Frequenzen 120 = 480/120 = 4
        if (y >= 0) {
            let freq = Math.trunc(y/4);
            startNoteFreq(freq, 127);
            oldFreq = freq;
        } else {
            oldFreq = -1;  // oldFreq = -1  --> kein Ton wird gespielt
        }
        // lfoGain.gain.value = 1-(y/HEIGHT);
        // console.log("masterGain.gain.value: ", masterGain.gain.value)

        break;
    case 240:  // sysex time (execution time)
        //execution-time
        var timeDelta = event.data[1];
        deltaCounter += 1;
        // add new value at the beginning
        deltaArray.unshift(timeDelta);
        // remove the last value
        deltaArray.pop();
    
        // console.log("timeDelta: ",  timeDelta)
        if (timeDelta < deltaMin){
            deltaMin = timeDelta
            document.getElementById("execution-min").innerHTML = deltaMin;
        }
        if (timeDelta > deltaMax){
            deltaMax = timeDelta
            document.getElementById("execution-max").innerHTML = deltaMax;
        }
        if (deltaCounter >= 100){
            deltaAvg = getDeltaAvg()
            document.getElementById("execution-avg").innerHTML = deltaAvg;
        }
        document.getElementById("execution-time").innerHTML = timeDelta;
        break;
    case 242:
        // IG: get X Value! And put corrresponding fx!
        var x = ((event.data[2] & 0b111111) << 7) + event.data[1];
        console.log("x: ", x)
        filter.frequency.value = ((x/WIDTH)*19080) +20;
        // console.log("filter.frequency.value: " + filter.frequency.value)

        // IG: panning according to x val
        factor = ((x/WIDTH) - 0.5)*2;
        //panning(factor);
        break; 
    }
}

// spiele Note einmalig ab
// startNote(1 + octaveShifter, 127);
startNoteFreq(59, 127);


function sendMiddleC( midiAccess, portID ) {
    var noteOnMessage = [0x90, 60, 0x7f];    // note on, middle C, full velocity
    var output = midiAccess.outputs.get(portID);
    output.send( noteOnMessage );  //omitting the timestamp means send immediately.
    output.send( [0x80, 60, 0x40], window.performance.now() + 1000.0 ); // Inlined array creation- note off, middle C,
                                                                        // release velocity = 64, timestamp = now + 1000ms.
  }
function sendMarkerChange(msg){
    if(output){
        if(msg=="marker_blue"){
            output.send([ 0x90, 0x45, 0x7f ]);            
            // console.log(output.send([B1,0001,0001]));
        }
        if(msg=="marker_green"){
            output.send(msg);
        }
        if(msg=="marker_red"){
            output.send(msg);
        }
        if(msg=="marker_yellow"){
            output.send(msg);
        }

    }
}

function loadSample(url) {
    return fetch(url)
      .then(response => response.arrayBuffer())
      .then(buffer => context.decodeAudioData(buffer));
    
  }

  function playSample(sample, note, velocity) {
    const source = context.createBufferSource();
    const gainTemp = context.createGain();
    source.buffer = sample;
    gainTemp.gain.linearRampToValueAtTime(0.05 + (0.33 * (velocity/127)), context.currentTime + attack)
    source.playbackRate.value = 2 ** ((note - 60) / 12);
    // source.connect(masterGain);
    source.connect(gainTemp);
    //attack
    // masterGain.connect(context.destination);
    gainTemp.connect(context.destination);
    // masterGain.gain.linearRampToValueAtTime(0.05 + (0.33 * (velocity/127)), context.currentTime + attack)


    source.start(0);
  }

  function stopSample(){
    masterGain.gain.cancelScheduledValues(0);
    masterGain.gain.linearRampToValueAtTime(0, context.currentTime + attack + release);
    source.stop(context.currentTime + attack + release + 0.1);
    source.disconnect();
  }


  // IG:
  function panning(factor)
  // factor: -1 (L) bis 1 (R) 0 = Stereo
  {
    // console.log("panning: factor: ", factor)
    //Init AudioContext
    // window.audioContext = window.audioContext||window.webkitAudioContext; //fallback for older chrome browsers
    // var context = new AudioContext();
    panNode.pan.setValueAtTime(factor, context.currentTime);
  }


  function listInputsAndOutputs( midiAccess ) {
    for (var entry of midiAccess.inputs) {
      var input = entry[1];
      console.log( "Input port [type:'" + input.type + "'] id:'" + input.id +
        "' manufacturer:'" + input.manufacturer + "' name:'" + input.name +
        "' version:'" + input.version + "'" );
    }
  
    for (var entry of midiAccess.outputs) {
      var output = entry[1];
      console.log( "Output port [type:'" + output.type + "'] id:'" + output.id +
        "' manufacturer:'" + output.manufacturer + "' name:'" + output.name +
        "' version:'" + output.version + "'" );
    }
  }



// https://stackoverflow.com/questions/30047056/is-it-possible-to-check-if-the-user-has-a-camera-and-microphone-and-if-the-permi
if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    // Firefox 38+ seems having support of enumerateDevicesx
    navigator.enumerateDevices = function(callback) {
        navigator.mediaDevices.enumerateDevices().then(callback);
    };
}

var MediaDevices = [];
var isHTTPs = location.protocol === 'https:';
var canEnumerate = false;

if (typeof MediaStreamTrack !== 'undefined' && 'getSources' in MediaStreamTrack) {
    canEnumerate = true;
} else if (navigator.mediaDevices && !!navigator.mediaDevices.enumerateDevices) {
    canEnumerate = true;
}

var hasMicrophone = false;
var hasSpeakers = false;
var hasWebcam = false;

var isMicrophoneAlreadyCaptured = false;
var isWebcamAlreadyCaptured = false;

function checkDeviceSupport(callback) {
    if (!canEnumerate) {
        return;
    }

    if (!navigator.enumerateDevices && window.MediaStreamTrack && window.MediaStreamTrack.getSources) {
        navigator.enumerateDevices = window.MediaStreamTrack.getSources.bind(window.MediaStreamTrack);
    }

    if (!navigator.enumerateDevices && navigator.enumerateDevices) {
        navigator.enumerateDevices = navigator.enumerateDevices.bind(navigator);
    }

    if (!navigator.enumerateDevices) {
        if (callback) {
            callback();
        }
        return;
    }

    MediaDevices = [];
    navigator.enumerateDevices(function(devices) {
        devices.forEach(function(_device) {
            var device = {};
            for (var d in _device) {
                device[d] = _device[d];
            }

            if (device.kind === 'audio') {
                device.kind = 'audioinput';
            }

            if (device.kind === 'video') {
                device.kind = 'videoinput';
            }

            var skip;
            MediaDevices.forEach(function(d) {
                if (d.id === device.id && d.kind === device.kind) {
                    skip = true;
                }
            });

            if (skip) {
                return;
            }

            if (!device.deviceId) {
                device.deviceId = device.id;
            }

            if (!device.id) {
                device.id = device.deviceId;
            }

            if (!device.label) {
                device.label = 'Please invoke getUserMedia once.';
                if (!isHTTPs) {
                    device.label = 'HTTPs is required to get label of this ' + device.kind + ' device.';
                }
            } else {
                if (device.kind === 'videoinput' && !isWebcamAlreadyCaptured) {
                    isWebcamAlreadyCaptured = true;
                }

                if (device.kind === 'audioinput' && !isMicrophoneAlreadyCaptured) {
                    isMicrophoneAlreadyCaptured = true;
                }
            }

            if (device.kind === 'audioinput') {
                hasMicrophone = true;
            }

            if (device.kind === 'audiooutput') {
                hasSpeakers = true;
            }

            if (device.kind === 'videoinput') {
                hasWebcam = true;
            }

            // there is no 'videoouput' in the spec.

            MediaDevices.push(device);
        });

        if (callback) {
            callback();
        }
    });
}

/////////////// WEBCAM CHECK

// src: https://masteringjs.io/tutorials/fundamentals/wait-1-second-then#:~:text=To%20delay%20a%20function%20execution,call%20fn%20after%201%20second.
function delay(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}

// check camera support!
async function checkWebcam() {
    await delay(5000);
    checkDeviceSupport(function() {
        document.write('hasWebCam: ', hasWebcam, '<br>');
        document.write('isWebcamAlreadyCaptured: ', isWebcamAlreadyCaptured, '<br>');
    });
    if (!hasWebcam){
        alert("Bitte eine Webcam anschließen!")
    }

    if (isWebcamAlreadyCaptured){
        alert("Deine Webcam wird bereits verwendet. Schließe bitte die andere Anwendung!")
    }
    navigator.mediaDevices.getUserMedia({ audio: false, video: true})
       .then(function (stream) {
             if (stream.getVideoTracks().length == 0) {
                 alert("Gib bitte Deine Webcam frei!")
             }
       })
      .catch(function (error) {
           // code for when there is an error
      });
}

// src: https://stackoverflow.com/questions/9899372/pure-javascript-equivalent-of-jquerys-ready-how-to-call-a-function-when-t/30757781#30757781
function ready(fn) {
  if (document.readyState != 'loading'){
    fn();
  } else if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', fn);
  } else {
    document.attachEvent('onreadystatechange', function() {
      if (document.readyState != 'loading')
        fn();
    });
  }
}

// ready(function() { checkWebcam(); });

