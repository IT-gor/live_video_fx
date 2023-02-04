
let markers = document.getElementsByClassName("select-marker");
let markers2 = document.getElementsByClassName("select-marker2");

let clearCanvas = document.getElementById("clearCanvas");

let marker = "marker_yellow";
let marker2 = "marker2_red";

let canvas_empty = true;

const colorspaceList = document.getElementById("colorspaceList");
const colorspaceSelect = document.getElementById("colorspaceSelect");


markers[0].classList.add('active')
// markers2[1].classList.add('active')

toggle(markers);
toggle(markers2);

// aus midi.js kopiert

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


function toggle(array){
    for(let i=0;i<array.length;i++){
        let imgSource= "images/"+array[i].getAttribute('value');
        if(array[i].classList.contains('active')){
            array[i].style.backgroundImage = `url(${imgSource}_color.png)`;
        }

        array[i].addEventListener("mousedown", function(){

            deToggleAll(array);
            array[i].classList.add('active');

            if(array[i].classList.contains('active')){
                array[i].style.backgroundImage = `url(${imgSource}_color.png)`;
            }

            if(array[i].hasAttribute('midi')){
                output = getOutput();
                output.send(JSON.parse(array[i].getAttribute('midi')));
            }

        });
        array[i].addEventListener("mouseover", function() {
            let imgSource= "images/"+array[i].getAttribute('value');
            array[i].style.backgroundImage = `url(${imgSource}_color.png)`;
        });
        array[i].addEventListener("mouseleave", function() {
            if(!array[i].classList.contains('active')){
                let imgSource= "images/"+array[i].getAttribute('value');
                array[i].style.backgroundImage = `url(${imgSource}.png)`;
            }
            
        });
    }
}

function deToggleAll(array){
    for(let i=0;i<array.length;i++){
        let imgSource= "images/"+array[i].getAttribute('value');
        array[i].classList.remove('active');
        array[i].style.backgroundImage = `url(${imgSource}.png)`;
    }
}
clearCanvas.addEventListener('click',function(e){
    output = getOutput();
    if (!canvas_empty) {
        clearCanvas.classList.remove('activeButton');
        output.send([177, 10, 9]);
        canvas_empty = true
    }

});

document.getElementById("btn-update").addEventListener("click", function() {
    var checkboxes = document.querySelectorAll('input[type="checkbox"]');
    var checked_values = [];
    for (var i = 0; i < checkboxes.length; i++) {
      if (checkboxes[i].checked) {
        checked_values.push(Number(checkboxes[i].value));
        console.log(checkboxes[i].value)
      }
    }
    let unchecked_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    unchecked_values = unchecked_values.filter(function(value) {
        return !checked_values.includes(value);
      });

    // setze deltamin und deltaMax zurück
    deltaMin = 128;
    deltaMax = 0;
    deltaCounter = 0;
    deltaArray = new Array(100).fill(0);
    document.getElementById("execution-min").innerHTML = deltaMin;
    document.getElementById("execution-max").innerHTML = deltaMin;
    output = getOutput();
    // first turns off all unchecked FX afterwards turns on all checked fx
    for (var i = 0; i < unchecked_values.length; i++) {
        output.send([177, 9, i]);
    }
    for (var i = 0; i < checked_values.length; i++) {
        output.send([177, 8, checked_values[i]]);
    }
    // send colorspace value if colorspace is chosen
    if (checked_values.includes(10)) { 
        var selectedValue = colorspaceSelect.options[colorspaceSelect.selectedIndex].value;
        output.send([177, 7, selectedValue]);
    }
  });

const checkbox = document.getElementById("checkbox10");

checkbox.addEventListener("change", function() {
if (checkbox.checked) {
    colorspaceList.style.display = "block";
} else {
    colorspaceList.style.display = "none";
}
});