import pandas as pd
import seaborn as sns
import matplotlib
# Import libs to solve classification task
from catboost import CatBoostClassifier


BEST_THRESHOLD = 0.5900894414440353
PATH_TO_MODEL = './models/my_catboost_model.cbm'


def get_catboost_model(path):
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = get_catboost_model(PATH_TO_MODEL)

    # Define optimal threshold
    model_th = BEST_THRESHOLD

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission


def make_feature_importance_json():
    model = get_catboost_model(PATH_TO_MODEL)
    features_scores = model.get_feature_importance(prettified=True).values
    return dict(features_scores[:5])


def make_dist_plot(submission):
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    sns_plot = sns.kdeplot(submission['preds'], fill=True)
    matplotlib.pyplot.title('Плотность распределения предсказаний модели')
    matplotlib.pyplot.xlabel('Предсказания')
    matplotlib.pyplot.ylabel('Плотность')
    fig = sns_plot.get_figure()
    return fig
