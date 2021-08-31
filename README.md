-Bert fox meme
1. Offensive speech classification pre-trained on fox news dataset
2. load fox news and fine tuned meme pretrained weights of bert

-context classifier
1. Tokenize and pad the sentences and fetch data, labels and dictionaries
2. Summarize the history for accuracy and loss

-create context data
1.merge politics output with sentences, find person from text and merge with sentence as new_sentence
2 send sentence for similarity and get context and save context in df
4 train model using context and sentence
5 store results
6 do ensemble

-effnet image classifier multioff
load data and 
train, predict model and
run model

-effnet image classifier politics

-get similar sentences
1.checks cosine similarity between given sentence and the wikipedia corpus and returns top k similar sentences
2.create model for generating embeddings and read context data and get the embeddings of context

-ML algo multiOFF text
1. load data and train models
2. predict according to the model


Execution steps
#run model

Data
multioff
fox meme
politics


-Utility files
1. Download google images: Creates a dataset by reading image links obtained from google images through 'Image Link Grabber' plugin and
 downloads each image for those links. Images are stored in data directory under their respective class folder.
2. Download wikipedia contents
3. fetch classification reports from outputs
4. Renaming multiOFF files
5. Renaming politics files
6. reads and re-formats MultiOFF dataset files to suit the efficientnet classifier dataset folder structure requirement
