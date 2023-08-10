from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Embedding, Flatten, SimpleRNN
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from sklearn.model_selection import train_test_split



def main():
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 50
    nlp = English()
    sentencizer = Sentencizer()
    np.set_printoptions(precision=2)

    with open("CFQ.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        CFQ = [ span.text.strip() for span in b.sents ]

    with open("E.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        E = [ span.text.strip() for span in b.sents ]

    with open("FC.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        FC = [ span.text.strip() for span in b.sents ]
    with open("HD.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        HD = [ span.text.strip() for span in b.sents ]
    with open("HH.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        HH = [ span.text.strip() for span in b.sents ]
    with open("IM.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        IM = [ span.text.strip() for span in b.sents ]
    with open("IR.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        IR = [ span.text.strip() for span in b.sents ]
    with open("J.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        J = [ span.text.strip() for span in b.sents ]
    with open("O.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        O = [ span.text.strip() for span in b.sents ]
    with open("WEU.txt", encoding="utf-8") as f:
        a = re.sub(r'\d+','',f.read())
        b = sentencizer(nlp(a))
        WEU = [ span.text.strip() for span in b.sents ]
   
    class_labels = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Class 1
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Class 2
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Class 3
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Class 4
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Class 5
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Class 6
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Class 7
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Class 8
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Class 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Class 10
    ]

    labels = np.concatenate([  
    np.repeat(np.array(class_labels[0]).reshape((1, 10)), len(CFQ), axis=0),
    np.repeat(np.array(class_labels[1]).reshape((1, 10)), len(E), axis=0),
    np.repeat(np.array(class_labels[2]).reshape((1, 10)), len(FC), axis=0),
    np.repeat(np.array(class_labels[3]).reshape((1, 10)), len(HD), axis=0),
    np.repeat(np.array(class_labels[4]).reshape((1, 10)), len(HH), axis=0),
    np.repeat(np.array(class_labels[5]).reshape((1, 10)), len(IM), axis=0),
    np.repeat(np.array(class_labels[6]).reshape((1, 10)), len(IR), axis=0),
    np.repeat(np.array(class_labels[7]).reshape((1, 10)), len(J), axis=0),
    np.repeat(np.array(class_labels[8]).reshape((1, 10)), len(O), axis=0),
    np.repeat(np.array(class_labels[9]).reshape((1, 10)), len(WEU), axis=0),
    ])


    allSents = CFQ + E + FC + HD + HH + IM + IR + J + O + WEU

    train_data, test_data, train_labels, test_labels = train_test_split(allSents, labels, test_size=0.2)

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    train_sequences = tokenizer.texts_to_sequences(train_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    # Pad the sequences to the same length
    train_sequences = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequences = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Define the model architecture
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=EMBEDDING_DIM)(input_layer)
    #recurrent nerual net
    rnn_layer = SimpleRNN(units=32)(embedding_layer)
    output_layer = Dense(units=10, activation='softmax')(rnn_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(train_sequences, train_labels, epochs=5, batch_size=32)
    

    # index to category dictionary
    dict = { 0 : "Consumer Financial Questions", 1 : "Education" , 2 : "Family & Children", 3 : "Health & Disability",
            4 : "Housing & Homelessness", 5 : "Income Maintenance", 6 : "Individual Rights", 7 : "Juvenile",
            8 : "Other", 9 : "Work, Employment, Unemployment" }

    sent = input("Enter a question: ")
    print()
    while (sent != 'done'):
        encoded_sent = tokenizer.texts_to_sequences([sent])
        padded_sent = pad_sequences(encoded_sent, 100)
        prediction = model.predict(padded_sent)
        #predicted_label = labels[np.argmax(prediction)]
        #print(prediction)
        print()

        total = np.sum(prediction[0])

        first = np.argmax(prediction[0])
        first_val = np.max(prediction[0])
        print(f"1. {dict.get(first)} - {round(first_val/total * 100, 2)}%")
        #print(f"1. {dict.get(first)} - {first * 100}%")
        prediction[0][first] = -1

        second = np.argmax(prediction[0])
        second_val = np.max(prediction[0])
        print(f"2. {dict.get(second)} - {round(second_val/total * 100, 2)}%")
        #print(f"2. {dict.get(second)} - {second * 100}%")
        prediction[0][second] = -1

        third = np.argmax(prediction[0])
        third_val = np.max(prediction[0])
        print(f"3. {dict.get(third)} - {round(third_val/total * 100, 2)}%")
        #print(f"3. {dict.get(third)} - {third * 100}%")
        prediction[0][third] = -1

        #print(prediction[0].index(max(prediction(0))))
        print()
        sent = input("Enter a question: ")

   
    """
    with open("CFQ.txt", "w") as f:
        cfq = categories["Consumer Financial Questions"]
        f.write(cfq)

    with open("E.txt", "w") as f:
        f.write(categories["Education"])

    with open("WEU.txt", "w") as f:
        f.write(categories["Work, Employment and Unemployment"])

    with open("FC.txt", "w") as f:
        f.write(categories["Family and Children"])

    with open("HD.txt", "w") as f:
        f.write(categories["Health and Disability"])

    with open("J.txt", "w") as f:
        f.write(categories["Juvenile"])

    with open("HH.txt", "w") as f:
        f.write(categories["Housing and Homelessness"])

    with open("IM.txt", "w") as f:
        f.write(categories["Income Maintenance"])

    with open("IR.txt", "w") as f:
        f.write(categories["Individual Rights"])

    with open("O.txt", "w") as f:
        f.write(categories["Other"])
"""

if __name__ == "__main__":
    main()

