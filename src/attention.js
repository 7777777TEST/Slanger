var App=App||{}
App.segmenter=new TinySegmenter()

App.Translator=class{
	async init(urls) {
		const translationMetadata = await (async (url)=>{try {
			const metadataJson = await fetch(url);
			const metadata = await metadataJson.json();
			console.log('Done loading metadata.');
			return metadata;
		} catch (err) {
			console.error(err);
			console.error('Loading metadata failed.');
		}})(urls.metadata);
		this.maxDecoderSeqLength = translationMetadata['max_decoder_seq_length'];
		this.maxEncoderSeqLength = translationMetadata['max_encoder_seq_length'];
		console.log('maxDecoderSeqLength = ' + this.maxDecoderSeqLength);
		console.log('maxEncoderSeqLength = ' + this.maxEncoderSeqLength);
		this.inputTokenIndex = translationMetadata['input_word_index'];
		this.targetTokenIndex = translationMetadata['target_word_index'];
		this.reverseTargetCharIndex = Object.keys(this.targetTokenIndex).reduce((obj, key) => (obj[this.targetTokenIndex[key]] = key, obj), {});
		this.model=await (async (url)=>{
			try {
				const model = await tf.loadLayersModel(url);
				console.log('Done loading pretrained model.');
				return model;
			} catch (err) {
				console.error(err);
				console.error('Loading pretrained model failed.');
				throw err;
			}
		})(urls.model)
	}
	preprocess_sentense(str=""){
		str=str.toLowerCase()
		var words=App.segmenter.segment(str)
		str="<start> ";
		words.forEach((s,n,a)=>{
			if(s!=" " && s!="   "){
				s=s.trim();
				str+=s+" ";
			}
		})
		str+="<end>"
		return str;
	}
	transform(encoding,data,vec_size){
		let transformed_data=tf.buffer([data.length,vec_size]);
		for(let i=0;i<data.length;i++){
			for(let j=0;j<Math.min(vec_size,data[i].split(" ").length);j++){
				if(data[i].split(" ")[j]=="")continue;
				console.log(data[i].split(" ")[j])
				transformed_data.set(encoding[data[i].split(" ")[j]],i,j);
			}
		}
		return transformed_data.toTensor();
	}
	seq2seq(str) {
		return tf.tidy(()=>{
			str=this.preprocess_sentense(str);
			console.log(str);
			let encoder_input=this.transform(this.inputTokenIndex,[str],this.maxEncoderSeqLength)
			console.log(encoder_input.arraySync())
			let decoder_input=tf.buffer([1,this.maxDecoderSeqLength])
			decoder_input.set(this.targetTokenIndex["<start>"],0,0)
			for (var i=1;i<this.maxEncoderSeqLength;i++){
				const predictOut=this.model.predict([encoder_input,decoder_input.toTensor()]);
				const output=predictOut.argMax(2).dataSync()[i-1]
				predictOut.dispose()
				decoder_input.set(output,0,i)
			}
			let output=""
			let final_output=this.model.predict([encoder_input,decoder_input.toTensor()]);
			final_output=final_output.argMax(2).dataSync()[this.maxDecoderSeqLength];
			for(let i=1;i<decoder_input.shape[1];i++){
				const sample_char=this.reverseTargetCharIndex[decoder_input.get(0,i)];
				if(sample_char=="<end>")break;
				output+=sample_char+" ";
			}
			return output;
		})

	}
	translate(inputSentence) {
		let decodedSentence="";
		decodedSentence=this.seq2seq(inputSentence)
		return decodedSentence;
	}
}