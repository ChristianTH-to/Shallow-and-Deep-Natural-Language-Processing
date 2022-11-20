# Using l-lda to seed topic output

# Relative
import model.labeled_lda as llda


# Document topics and seeds
labeled_documents = [("never_forget cop holy_crap holy_fuck lol", ["true_god"]),
                     ("smoke drugs art color bong based sound lit", ["weed"]),
                     ("cat awesome my_wife gold austin_powers max_mojo", ["million_dollars"]),
                     ("believe feel love dream fun looking_forward trust", ["friend"]),
                     ("plus die kill awful working hate dead job", ["negative, death"])]

# New Labeled LDA model
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector="50_div_K", eta_vector=0.001)
# llda_model = llda.LldaModel(labeled_documents=labeled_documents, alpha_vector=0.02, eta_vector=0.002)
llda_model = llda.LldaModel(labeled_documents=labeled_documents)
print llda_model


# Train
# llda_model.training(iteration=10, log=True)
while True:

    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)

    print "after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity)
    if llda_model.is_convergent:

        break

        
# Update
print "before updating: ", llda_model

update_labeled_documents = [("cat awesome my_wife gold austin_powers max_mojo", ["million_dollars"])]
llda_model.update(labeled_documents=update_labeled_documents)
print "after updating: ", llda_model


# Train again
# llda_model.training(iteration=10, log=True)
while True:

    print("iteration %s sampling..." % (llda_model.iteration + 1))

    llda_model.training(1)

    print "after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity)
    if llda_model.is_convergent:

        break

        
# Inference
# note: the resulting topics may be different from training, because gibbs sampling is a random algorithm
document = "believe feel love dream fun looking_forward trust plus die kill awful working hate dead job"
# topics = llda_model.inference(document=document, iteration=10, times=10)
topics = llda_model.inference_multi_processors(document=document, iteration=10, times=10)
print topics

# Save to disk
save_model_dir = "../data/model"
# llda_model.save_model_to_dir(save_model_dir, save_derivative_properties=True)
llda_model.save_model_to_dir(save_model_dir)

# Load from disk

llda_model_new = llda.LldaModel()
llda_model_new.load_model_from_dir(save_model_dir)
print "llda_model_new", llda_model_new
print "llda_model", llda_model
