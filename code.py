import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import numpy as np

file_path = '/content/train.tsv'
data = pd.read_csv(file_path, sep='\t')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length',
            truncation=True
        )

        input_ids.append(input_dict["input_ids"])
        attention_masks.append(input_dict["attention_mask"])
        token_type_ids.append(input_dict["token_type_ids"])
        labels.append(e.label)

    return (
        tf.convert_to_tensor(input_ids),
        tf.convert_to_tensor(attention_masks),
        tf.convert_to_tensor(token_type_ids)
    ), tf.one_hot(labels, depth=3)


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_data['sentiment']), y=train_data['sentiment'])
class_weights_dict = {i: class_weights[i] for i in range(3)}




train_input, train_labels = convert_examples_to_tf_dataset(
    list(train_data.apply(lambda x: InputExample(guid=None, text_a=x['text'], text_b=None, label=x['sentiment']), axis=1)),
    tokenizer
)
val_input, val_labels = convert_examples_to_tf_dataset(
    list(val_data.apply(lambda x: InputExample(guid=None, text_a=x['text'], text_b=None, label=x['sentiment']), axis=1)),
    tokenizer
)



model = TFBertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels=3)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])



model.fit(train_input, train_labels, epochs=3, batch_size=32, validation_data=(val_input, val_labels), class_weight=class_weights_dict)


val_pred_probs = model.predict(val_input)
val_logits = val_pred_probs.logits
val_preds = tf.argmax(val_logits, axis=1)

f1 = f1_score(tf.argmax(val_labels, axis=1), val_preds, average='macro')
print(f'Macro F1 Score: {f1}')


model.save('/content/sentiment_analysis_model')
tokenizer.save_pretrained('/content/sentiment_analysis_model')
model.save_pretrained('/content/sentiment_analysis_model')

