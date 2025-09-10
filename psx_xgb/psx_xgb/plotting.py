import pandas as pd
import matplotlib.pyplot as plt


def save_feature_importance(model, features: list[str], out_path: str = 'feature_importance.png'):
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    ax = fi.plot(kind='barh', figsize=(10, max(4, len(features) * 0.35)))
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


