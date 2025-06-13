from src.config.config import load_config
from src.data.data_loader import load_data
from src.data.preprocess import preprocess_data
from src.models.train_model import train_model, save_model
from src.models.evaluate_model import evaluate_model

def run_training():
    config = load_config()
    df = load_data(config['data_path'])
    X_train, X_test, y_train, y_test = preprocess_data(df, config['target_column'])
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, config['model_output'])
