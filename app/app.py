import os
import json

from flask import (Flask, request,
                   render_template, redirect,
                   url_for, send_from_directory)
from werkzeug.utils import secure_filename
from datetime import datetime
# Import modelling scripts
import src.preprocessing as preprocessing
import src.scorer as scorer


ALLOWED_EXTENSIONS = set(['csv'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    app = Flask(__name__)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':

            # Import file
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                new_filename = f'{filename.split(".")[0]}_\
                    {str(datetime.now())}.csv'
                new_filename = secure_filename(new_filename)

                # Store imported file locally
                save_location = os.path.join('input', new_filename)
                file.save(save_location)

                # Get input dataframe
                input_df = preprocessing.import_data(save_location)

                # Run preprocessing
                preprocessed_df = preprocessing.run_preproc(input_df)

                # Run scorer to get submission file for competition
                submission = scorer.make_pred(preprocessed_df, save_location)
                submission.to_csv(
                    save_location.replace('input', 'output'), index=False
                )

                top_5_feature_importance = scorer.make_feature_importance_json()
                download_feature_location = os.path.join(
                    'output', 'top_5_feature_importance.json'
                )
                with open(download_feature_location, mode='w') as f:
                    json.dump(top_5_feature_importance, f, ensure_ascii=False)

                predictions_dist_plot_name = os.path.join(
                    'output', 'dist_plot_.png'
                )
                fig = scorer.make_dist_plot(submission)
                fig.savefig(predictions_dist_plot_name)
                return redirect(url_for('download'))

        return render_template('upload.html')

    @app.route('/download')
    def download():
        return render_template('download.html', files=os.listdir('output'))

    @app.route('/download/<filename>')
    def download_file(filename):
        return send_from_directory('output', filename)

    return app
