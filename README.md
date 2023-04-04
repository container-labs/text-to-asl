# text-to-asl
A transformer build to take text and display American Sign Language

## Summary

Developing a text-to-video transformer for ASL is a challenging and ambitious project, but it has the potential to make a significant impact on the accessibility of communication for the Deaf community. Anyone who wants to help is welcome to.

## Strategy

Since sign language is a visual language that relies on movement and facial expressions, a text-to-video transformer would be more suitable for capturing the dynamics of ASL. This would enable the generation of more accurate and natural sign language representations from text input.

To create a text-to-video transformer, roughly:

Data preparation: Use the video datasets (ASL Lexicon Video Dataset, RWTH-BOSTON-104 Database, or others). We will need to preprocess these datasets to extract relevant information such as text labels, video frames, and time information.

Video feature extraction: Use pre-trained models like 3D CNNs (e.g., I3D, C3D, or TSN) to extract spatiotemporal features from the video frames. These features will serve as input for the transformer model.

Text feature extraction: Process the text input using a pre-trained language model, such as BERT or GPT, to obtain the text embeddings. This will help the model understand the context and semantics of the input text.

Transformer architecture: Design a transformer model that takes both video and text features as input and generates video sequences as output. Explore variations of popular transformer architectures, such as the original Transformer, BERT, or GPT, to find a suitable model for the task.

Training and fine-tuning: Train the model on the preprocessed dataset, using an appropriate loss function and optimization strategy. We might need to fine-tune the pre-trained video and text models on the task to achieve better performance.

Evaluation and improvement: Evaluate the model using relevant metrics, such as accuracy, F1-score, or BLEU score, to measure the quality of the generated videos. Use this feedback to iteratively improve the model's architecture, training strategy, or data augmentation techniques.



### Resources 

+ Labeled ASL alphabet data [link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
