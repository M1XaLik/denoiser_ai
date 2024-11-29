import os
import logging
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from denoiser import denoise_image

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    file = FileField('Upload Image', validators=[DataRequired()])
    submit = SubmitField('Submit')

# INIT APP
app = Flask(__name__)

# SECRET KEY FOR SIGNING DATA STORED IN SESSIONS AND COOKIES
SECRET_KEY_variable = os.urandom(32)

# APP CONFIG
app.config.update(
    SECRET_KEY=SECRET_KEY_variable,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_DATABASE_URI='sqlite:///db.sqlite',
    WTF_CSRF_ENABLED=True,
    WTF_CSRF_SECRET_KEY=SECRET_KEY_variable,
    UPLOAD_FOLDER='static/uploads',
    PROCESSED_FOLDER='static/processed'
)

# Create directories if they don't exist 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# INIT DATABASE
db = SQLAlchemy(app)
migrate = Migrate(app=app, db=db)  # and migrations

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from datetime import datetime

#IMAGE TABLE
class ImageModel(db.Model):
    __tablename__ = "images"
    id = db.Column(db.Integer, primary_key=True)
    file_url = db.Column(db.String(200), nullable=False)
    uploader = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) 

    def __repr__(self):
        return f"<Image record id={self.id}, file_url={self.file_url}, uploader={self.uploader}, timestamp={self.timestamp}>"


@app.route('/', methods=['GET', 'POST'])
def index():
    logger.debug("Rendering index page")
    form = UploadForm()

    if form.validate_on_submit():
        file = form.file.data
        if file:
            # GENERATE A UNIQUE IDENTIFIER
            unique_id = str(uuid.uuid4())
            original_filename = secure_filename(file.filename)

            # NAME FOR THE ORIGINAL FILE
            filename = f"{unique_id}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.debug(f"File uploaded: {filename}")

            # PROCESS THE IMAGE AND GET THE PROCESSED FILE NAME
            processed_filename = process_image(file_path)

            # SAVE ONLY FILE NAMES TO THE DATABASE
            original_image_record = ImageModel(file_url=filename, uploader='user')
            processed_image_record = ImageModel(file_url=processed_filename, uploader='bot')

            db.session.add(original_image_record)
            db.session.add(processed_image_record)
            db.session.commit()
            logger.debug(f"Image records saved to database: {original_image_record}, {processed_image_record}")

    # RETRIEVE ALL RECORDS FROM THE DATABASE, SORTED BY TIMESTAMP DESCENDING
    images = ImageModel.query.order_by(ImageModel.timestamp.desc()).all()
    logger.debug(f"Images retrieved from database: {images}")

    return render_template('index.html', form=form, images=images)

def process_image(file_path):
    try:
        logger.debug(f"Processing image: {file_path}")

        # NAME FOR THE PROCESSED FILE
        processed_filename = f"processed_{os.path.basename(file_path)}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        # CALL THE IMAGE PROCESSING FUNCTION FROM `DENOISER`
        denoise_image(file_path, processed_path)
        logger.debug(f"Image processed: {processed_path}")

        return processed_filename
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

if __name__ == '__main__':
    logger.debug("Starting Flask application")
    app.run(debug=True)
