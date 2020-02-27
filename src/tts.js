var App=App||{}
// App.TTS=(text,lang)=>{
// 	var msg=new window.SpeechSynthesisUtterance()
// 	var voices=window.speechSynthesis.getVoices()
// 	msg.volume=1.0
// 	msg.rate=1.0
// 	msg.text=text
// 	window.speechSynthesis.speak(msg)
// }
App.TTS=async (text,lang,access_token)=>{

	async function synthesis(access_token, input_lang, text) {
		const base_url = "https://sandbox-ss.mimi.fd.ai/speech_synthesis";
		const form_data = new FormData();
		form_data.append("lang", input_lang);
		form_data.append("text", text);
		form_data.append("engine", "nict");

		return await fetch(base_url, {
			method: "POST",
			mode: "cors",
			headers: {
				"Authorization": "Bearer " + access_token
			},
			body: form_data,
		})
		.then(response => response.arrayBuffer())
	}
	if (text.length < 1) {
		throw new Error("Input text");
	}

	if (access_token == "") {
		var msg=new window.SpeechSynthesisUtterance()
		msg.volume=1.0
		msg.rate=1.0
		msg.text=text
		window.speechSynthesis.speak(msg)
		return;
	}

	let input_lang = lang;

	const context = new window.AudioContext();
	const source = context.createBufferSource();

	await synthesis(access_token, input_lang, text)
		.then(ab => context.decodeAudioData(ab, buf => {
			source.buffer = buf;
			source.connect(context.destination);
			source.start();
		}))
		.catch(error => console.error("synthesis error"));
}