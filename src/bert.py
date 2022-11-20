from packages import *


class BertClassifier:
    def __init__(self, data, local_path, cuda):

        # get the data
        self.data = data.copy(deep=True)

        # Load a trained model and vocabulary that you have fine-tuned
        output_dir = local_path + "/../" + "models/bert/epoch_2"
        self.bert = BertForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir)

        # Copy the model to the GPU. Tell pytorch to run this model on the GPU.
        if cuda:
            self.bert.cuda()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)

    def predict(self):
        # Start testing
        self.data_preprocessing()
        self.start_test()

    def training(self):
        pass  # TODO

    def data_preprocessing(self):

        # Report the number of sentences.
        print("Number of test sentences: {:,}\n".format(self.data.shape[0]))

        # Create sentence and label lists
        emails = self.data["LastEmailContent"].values
        self.labels = self.data["Type"].values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        self.input_ids = []
        self.attention_masks = []

        # For every sentence...
        for email in tqdm(emails):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                email,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=128,  # Pad & truncate all sentences.
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            self.input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            self.attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.labels)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.
        prediction_data = TensorDataset(
            self.input_ids, self.attention_masks, self.labels
        )
        prediction_sampler = SequentialSampler(prediction_data)
        self.prediction_dataloader = DataLoader(
            prediction_data, sampler=prediction_sampler, batch_size=batch_size
        )

    def start_test(self):

        print("\n Running NLP (BERT) classification... \n")

        # Put model in evaluation mode
        self.bert.eval()

        # Tracking variables
        predictions, true_labels, probabilities = [], [], []

        # Predict
        for batch in tqdm(self.prediction_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.bert(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )

            logits = outputs[0]
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Store predictions and true labels
            predictions.append(logits)
            probabilities.append(probs)
            true_labels.append(label_ids)

        self.predictions = np.concatenate(probabilities, axis=0)[:, 1]

        print("\nNLP (BERT) classification done. \n")
