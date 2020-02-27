var App=App||{}
App.Translator=class{
	async init(urls) {
		const model=await (async (url)=>{
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
		this.inputTokenIndex = translationMetadata['input_token_index'];
		this.targetTokenIndex = translationMetadata['target_token_index'];
		this.reverseTargetCharIndex = Object.keys(this.targetTokenIndex).reduce((obj, key) => (obj[this.targetTokenIndex[key]] = key, obj), {});
		this.prepareEncoderModel(model)
		this.prepareDecoderModel(model)
	}
	prepareEncoderModel(model) {
		this.numEncoderTokens = model.input[0].shape[2];
		console.log('numEncoderTokens = ' + this.numEncoderTokens);
		const encoderInputs = model.input[0];
		const stateH = model.layers[2].output[1];
		const stateC = model.layers[2].output[2];
		const encoderStates = [stateH, stateC];

		this.encoderModel = tf.model({inputs: encoderInputs, outputs: encoderStates});
	}
	prepareDecoderModel(model) {
		this.numDecoderTokens = model.input[1].shape[2];
		console.log('numDecoderTokens = ' + this.numDecoderTokens);
		const stateH = model.layers[2].output[1];
		const latentDim = stateH.shape[stateH.shape.length - 1];
		console.log('latentDim = ' + latentDim);
		const decoderStateInputH = tf.input({shape: [latentDim], name: 'decoder_state_input_h'});
		const decoderStateInputC = tf.input({shape: [latentDim], name: 'decoder_state_input_c'});
		const decoderStateInputs = [decoderStateInputH, decoderStateInputC];
		const decoderLSTM = model.layers[3];
		const decoderInputs = decoderLSTM.input[0];
		const applyOutputs = decoderLSTM.apply(decoderInputs, {initialState: decoderStateInputs});
		let decoderOutputs = applyOutputs[0];
		const decoderStateH = applyOutputs[1];
		const decoderStateC = applyOutputs[2];
		const decoderStates = [decoderStateH, decoderStateC];
		const decoderDense = model.layers[4];
		decoderOutputs = decoderDense.apply(decoderOutputs);
		this.decoderModel = tf.model({inputs: [decoderInputs].concat(decoderStateInputs),outputs: [decoderOutputs].concat(decoderStates)});
	}
	encodeString(str) {
		const strLen = str.length;
		const encoded = tf.buffer([1, this.maxEncoderSeqLength, this.numEncoderTokens]);
		for (let i = 0; i < strLen; ++i) {
			if (i >= this.maxEncoderSeqLength) {
				console.error( 'Input sentence exceeds maximum encoder sequence length: ' + this.maxEncoderSeqLength);
			}

			const tokenIndex = this.inputTokenIndex[str[i]];
			if (tokenIndex == null) {
				console.error( 'Character not found in input token index: "' + tokenIndex + '"');
			}
			encoded.set(1, 0, i, tokenIndex);
		}
		return encoded.toTensor();
	}

	decodeSequence(inputSeq) {
		let statesValue = this.encoderModel.predict(inputSeq);

		let targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
		targetSeq.set(1, 0, 0, this.targetTokenIndex['\t']);

		let stopCondition = false;
		let decodedSentence = '';
		while (!stopCondition) {
			const predictOutputs = this.decoderModel.predict([targetSeq.toTensor()].concat(statesValue));
			const outputTokens = predictOutputs[0];
			const h = predictOutputs[1];
			const c = predictOutputs[2];

			const logits = outputTokens.reshape([outputTokens.shape[2]]);
			const sampledTokenIndex = logits.argMax().dataSync()[0];
			const sampledChar = this.reverseTargetCharIndex[sampledTokenIndex];
			decodedSentence += sampledChar;

			if (sampledChar === '\n' ||
					decodedSentence.length > this.maxDecoderSeqLength) {
				stopCondition = true;
			}

			targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
			targetSeq.set(1, 0, 0, sampledTokenIndex);

			statesValue = [h, c];
		}

		return decodedSentence;
	}

	translate(inputSentence) {
		let decodedSentence=""
		tf.tidy(()=>{
			const inputSeq = this.encodeString(inputSentence);
			decodedSentence = this.decodeSequence(inputSeq);
		})
		return decodedSentence;
	}
}