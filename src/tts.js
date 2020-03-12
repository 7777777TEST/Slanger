var App=App||{}
App.TTS=async (text,input_lang,access_token)=>{
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
	const context=new window.AudioContext();
	const source=context.createBufferSource();
	await synthesis(access_token,input_lang,text).then(ab=>{
		context.decodeAudioData(ab,buf=>{
			source.buffer=buf;
			source.connect(context.destination);
			source.start();
		}).catch(e=>{
			console.error("error:",e);
		})
	})
}