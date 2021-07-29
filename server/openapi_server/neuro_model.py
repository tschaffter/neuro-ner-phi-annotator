
from neuroner import neuromodel

nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True, dataset_text_folder="data/example_unannotated_texts", pretrained_model_folder ="trained_models/i2b2_2014_glove_spacy_bioes", 
parameters_filepath='.', spacylanguage="en_core_web_sm", load_all_pretrained_token_embeddings=1,maximum_number_of_epochs=10000)


result = nn.predict("""Dr. James and I live in Columbia Missouri. 
              My phone number is 573-529-5658
              Tom's email is clem_james@yahoo.com. 
              My favorite website is https://www.google.com. 
              My social security is 886125250. 
              I was born on January 1st, 1900.  
              Today is 07/09/2021.
          """
)

for record in result:
    print(record['text'], record['start'], record['end'], record['type'])

