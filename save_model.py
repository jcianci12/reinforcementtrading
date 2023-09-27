def save_model(model, model_name):
    # Save the model to disk
    model.save(f'{model_name}.h5')