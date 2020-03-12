var App=App||{};
App.baseURL="http://localhost/slanger/models/"
App.translator=new App.Translator();
App.input="eng"
App.output="jpn"
App.Trigger=()=>{
	var urls={model:App.baseURL+App.input+"-"+App.output+"/model.json",metadata:App.baseURL+App.input+"-"+App.output+"/metadada.json"}
	App.translator.init(urls)
	App.ASR(input);
	document.addEventListener("asr-get",e=>{
		var text=App.translator.translate(e.text)
		App.TTS(text,output)
		var talkroom=document.getElementById("talk")
		var list=document.createElement("li")
		list.textContent=text
		talkroom.appendChild(list)
	})
}
App.Trigger=(input,output)=>{
	var urls={model:App.baseURL+App.input+"-"+App.output+"/model.json",metadata:App.baseURL+App.input+"-"+App.output+"/metadada.json"}
	App.translator.init(urls)
	App.ASR(input,(text)=>{
		var text=App.translator.translate(text)
		var talkroom=document.getElementById("talk")
		var list=document.createElement("li")
		list.textContent=text
		talkroom.appendChild(list)
		App.TTS(text,output)
	});
}
App.Translate=()=>{
	var model_url=App.baseURL+App.input+"-"+App.output+"/model.json"
	var meta_url=App.baseURL+App.input+"-"+App.output+"/metadada.json"
	var urls={model:model_url,metadata:meta_url}
	console.log(urls)
	App.translator.init(urls).then(e=>{
		console.log("Loaded model")
		var textbox=document.getElementById("textbox");
		var text=App.translator.translate(textbox.value);
		var result=document.getElementById("result");
		console.log(text)
		result.value=text;
	},e=>{
		console.error(e)
	})
}
document.getElementById("translate").addEventListener("mousedown|touchend",App.Translate)
document.getElementById("reverse").addEventListener("mousedown|touchend",e=>{
	var input=App.output;
	App.output=App.input;
	App.input=input;
})