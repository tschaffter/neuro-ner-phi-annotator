
from neuroner import neuromodel

nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True, dataset_text_folder="data/example_unannotated_texts", pretrained_model_folder ="trained_models/i2b2_2014_glove_spacy_bioes", 
spacylanguage="en_core_web_sm")


a = ((nn.predict("""My name is Wolfgang and I live in Columbia Missouri . My phone number is
alternate no is 5731111111. Tom's email is wolfgang_00000@yahoo.com. My website is
www.stilt.com. My social security is 886-00-0000. I was born on 1st of January 1900. 
Today is 07/09/2021."""
)))

# print(dir(nn))
# print((a[0]))
#python -m spacy download en_core_web_sm

