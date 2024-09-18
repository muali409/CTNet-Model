import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Softmax
from tensorflow.keras.models import Model

#  Pre-trained CNN Modules
def build_cnn_model(input_shape, model_type="efficientnet"):
    if model_type == "efficientnet":
        base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_type == "densenet":
        base_model = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
    else:
        raise ValueError("Unsupported model type")

    base_model.trainable = False
    return base_model

#  Cross-Channel Attention Module
def cross_channel_attention(inputs):
    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)

    avg_pool = layers.Reshape((1, 1, -1))(avg_pool)
    max_pool = layers.Reshape((1, 1, -1))(max_pool)

    alpha = Dense(1, activation='relu')(avg_pool)
    beta = Dense(1, activation='relu')(max_pool)

    # Dynamic weighting
    alpha_prime = layers.Softmax()(alpha)
    beta_prime = layers.Softmax()(beta)

    # Feature fusion
    weighted_avg = layers.Multiply()([alpha_prime, avg_pool])
    weighted_max = layers.Multiply()([beta_prime, max_pool])

    fused = layers.Add()([weighted_avg, weighted_max])

    # Cross-channel attention
    attention_scores = Dense(inputs.shape[-1], activation='sigmoid')(fused)
    modulated_features = layers.Multiply()([inputs, attention_scores])

    return modulated_features

# Transformer Integration
def transformer_block(inputs, num_heads=4, ff_dim=32):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ffn_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)

    transformer_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    return transformer_output

# Feature Fusion and Classification
def build_ctnet_model(input_shape_mias, input_shape_breakhis, num_classes):
    # CNN feature extraction
    mammogram_cnn = build_cnn_model(input_shape_mias, model_type="efficientnet")
    histopathology_cnn = build_cnn_model(input_shape_breakhis, model_type="densenet")

    # Input layers for the two modalities
    input_mias = layers.Input(shape=input_shape_mias)
    input_breakhis = layers.Input(shape=input_shape_breakhis)

    # Extract features
    mias_features = mammogram_cnn(input_mias)
    breakhis_features = histopathology_cnn(input_breakhis)

    # Cross-channel attention
    mias_attention = cross_channel_attention(mias_features)
    breakhis_attention = cross_channel_attention(breakhis_features)

    # Transformer layers
    mias_transformed = transformer_block(mias_attention)
    breakhis_transformed = transformer_block(breakhis_attention)

    # Feature fusion
    fused_features = layers.Multiply()([mias_transformed, breakhis_transformed])

    # Classification head
    x = GlobalAveragePooling2D()(fused_features)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Build and compile model
    model = Model(inputs=[input_mias, input_breakhis], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Model training
def train_model(mias_data, breakhis_data, num_classes, epochs=50, batch_size=32):
    input_shape_mias = mias_data[0].shape[1:]
    input_shape_breakhis = breakhis_data[0].shape[1:]

    # Build the hybrid CTNet model
    model = build_ctnet_model(input_shape_mias, input_shape_breakhis, num_classes)

    # Train the model
    model.fit(
        [mias_data[0], breakhis_data[0]], 
        mias_data[1],  # Assume same labels for both datasets, or concatenate labels accordingly
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    return model
