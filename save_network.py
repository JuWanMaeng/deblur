from basicsr.models.archs.UFPNet_code_uncertainty_arch import UFPNet_code_uncertainty

# Save weights
model = UFPNet_code_uncertainty()



model.save_weights('model_weights.pth')


