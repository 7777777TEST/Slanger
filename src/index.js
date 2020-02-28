"use strict";
var App = App || {}
App.baseURL = "http://7777777TEST.github.io/Slanger/models/"
App.translator = new App.Translator();
App.output = "ja"
App.input = "en"

const time_interval = 300;
let mediaRecorder = null;
const base_url = "wss://sandbox-sr.mimi.fd.ai";
const button_rec_start = document.getElementById("button_rec_start");
const button_rec_stop = document.getElementById("button_rec_stop");
const donelist = document.getElementById("donelist");
App.speaker = 1;
const mylog = message => {
	const e = document.createElement("div");
	e.appendChild(document.createTextNode(message));
	if (App.speaker == 1) {
		e.className = "chatR";
	} else {
		e.className = "chatL";
	}
	donelist.insertBefore(e, donelist.firstChild);
};

App.Translate = (text) => {
	var model_url = App.baseURL + App.input + "-" + App.output + "/model.json"
	var meta_url = App.baseURL + App.input + "-" + App.output + "/metadada.json"
	var urls = { model: model_url, metadata: meta_url }
	console.log(urls)
	App.translator.init(urls).then(e => {
		mylog("Loaded model")
		text = App.translator.translate(text);
		mylog(text);
		App.TTS(text, App.output, document.getElementById("token").value)
	}, e => {
		mylog("Error")
	})
}

const connect = url => {
	const socket = new WebSocket(url);
	socket.onopen = event => {
		console.log("WebSocket open.");
		navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
			mediaRecorder = new MediaStreamRecorder(stream);
			mediaRecorder.audioChannels = 1;
			mediaRecorder.mimeType = 'audio/pcm';
			mediaRecorder.ondataavailable = blob => {
				socket.send(blob);
			};
			mediaRecorder.onstop = () => {
				socket.send('{"command":"recog-break"}');
			};
			mediaRecorder.start(time_interval);
		}).catch(e => {
			console.error('media error: ', e);
		});
	};
	socket.onmessage = event => {
		console.log("WebSocket message: " + event.data);
		App.result = event.data
		var res = JSON.parse(event.data);
		console.log(res);
		var text = "";
		for (var i = 0; i < res.response.length; i++) {
			text += res.response[i].result.split("|")[0] + " ";
		}
		text = text.trimEnd()
		text = text.charAt(0).toUpperCase() + text.slice(1) + "."
		console.log("REC:", text)
		App.Translate(text);
	};
	socket.onerror = event => {
		console.log("WebSocket message: " + event);
	};
	socket.onclose = event => {
		button_rec_start.disabled = false;
		button_rec_stop.disabled = true;
	};
};
document.getElementById("translate").onclick = (e) => {
	App.Translate(document.getElementById("textbox").value)
}
document.getElementById("reverse").onclick = (e) => {
	App.speaker *= -1;
	var out = App.input;
	App.input = App.output;
	App.output = out;
	if (App.speaker == 1) {
		document.getElementById("reverse").textContent = "⇐";
	} else {
		document.getElementById("reverse").textContent = "⇒";
	}
}
button_rec_start.onclick = event => {
	const access_token = document.getElementById("token").value;
	if (access_token == "") {
		mylog("ERROR: Specify access token.");
		throw new Error("Specify access token.");
	}
	button_rec_start.disabled = true;
	button_rec_stop.disabled = false;
	const url = base_url + "/?process=nict-asr&access-token=" + encodeURIComponent(access_token) + "&input-language=" + encodeURIComponent(App.input) + "&content-type=audio%2Fx-pcm%3Bbit%3D16%3Brate%3D44100%3Bchannels%3D1";
	connect(url);
};
button_rec_stop.onclick = event => {
	mediaRecorder.stop();
	button_rec_stop.disabled = true;
}
