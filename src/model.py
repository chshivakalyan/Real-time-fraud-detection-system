import lightgbm as lgb

def get_model(scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        scale_pos_weight=scale_pos_weight,
        device="gpu"   # uses RTX
    )
